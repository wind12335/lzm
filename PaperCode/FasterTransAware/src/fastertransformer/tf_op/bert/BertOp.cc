/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/bert/Bert.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Bert")
    .Input("from_tensor: T")
    .Input("to_tensor: T")
    .Input("sequence_length: int32")
    .Input("attr_q_kernel: N * T")
    .Input("attr_q_bias: N * T")
    .Input("attr_k_kernel: N * T")
    .Input("attr_k_bias: N * T")
    .Input("attr_v_kernel: N * T")
    .Input("attr_v_bias: N * T")
    .Input("attr_output_kernel: N * T")
    .Input("attr_output_bias: N * T")
    .Input("attr_output_layernorm_beta: N * T")
    .Input("attr_output_layernorm_gamma: N * T")
    .Input("inter_kernel: N * T")
    .Input("inter_bias: N * T")
    .Input("output_kernel: N * T")
    .Input("output_bias: N * T")
    .Input("output_layernorm_beta: N * T")
    .Input("output_layernorm_gamma: N * T")
    .Output("output: T")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("remove_padding: bool")
    .Attr("q_scaling: float")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return tf::Status::OK();
    });

template<typename T>
class TFTraits;

template<>
class TFTraits<float> {
public:
    typedef float DataType;
};

template<>
class TFTraits<Eigen::half> {
public:
    typedef half DataType;
};

#ifdef ENABLE_BF16
template<>
class TFTraits<tf::bfloat16> {
public:
    typedef __nv_bfloat16 DataType;
};
#endif

template<typename Device, typename T>
class BertOp: public BaseOp<T> {
public:
    explicit BertOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {   
        try {
            // 初始化参数并赋值
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
            OP_REQUIRES_OK(context, context->GetAttr("remove_padding", &remove_padding_));
            OP_REQUIRES_OK(context, context->GetAttr("q_scaling", &q_scaling_));
            // 获取流处理器（GPU）版本（就是获取其算力）
            sm_              = ft::getSMVersion();
            // FasterTransformer/src/fastertransformer/utils/cublasAlgoMap.cc
            cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");
        }
        catch (std::runtime_error& error) {
            OP_REQUIRES(context, false, tf::errors::Internal(error.what()));
        }
    }

    ~BertOp()
    {
        delete cublas_algo_map_;
    }

    void Compute(tf::OpKernelContext* context) override
    {   
        // REGISTER_OP里面input的数量, BERT是3(input, output, seq_len) + 12个layers * 16个参数每层layer
        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 16) + 3,
                    tf::errors::InvalidArgument("[ERROR] More or Less input arguments"));

        // batch_size是第一个输入的第一个维度
        // seq_len是第一个输入第二个维度
        const size_t batch_size_   = (size_t)context->input(0).dim_size(0);
        const size_t from_seq_len_ = (size_t)context->input(0).dim_size(1);

        // 输入和输出的batch必须相等
        OP_REQUIRES(context,
                    batch_size_ == (size_t)context->input(2).dim_size(0),
                    tf::errors::InvalidArgument("[ERROR] invalid shape"));

        // 构建流和申请空间
        const cudaStream_t& stream        = context->eigen_device<Device>().stream();
        cublasHandle_t      cublas_handle = this->get_cublas_handler();
        cublasSetStream(cublas_handle, stream);
        ft::Allocator<ft::AllocatorType::TF> allocator(context, stream);
        // gemm wrapper
        ft::cublasMMWrapper                  cublas_wrapper = ft::cublasMMWrapper(cublas_handle,
                                                                 this->get_cublaslt_handler(),
                                                                 stream,
                                                                 cublas_algo_map_,
                                                                 this->get_cublas_wrapper_mutex(),
                                                                 &allocator);
        

        // gemm参数配置
        if (std::is_same<T, Eigen::half>::value) {
            cublas_wrapper.setFP16GemmConfig();
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, tf::bfloat16>::value) {
            cublas_wrapper.setBF16GemmConfig();
        }
#endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        // 存储bert的权重
        ft::BertWeight<DataType> bert_weights;
        bert_weights.bert_layer_weights.resize(num_layer_);

        // 将输入的权重全部转移到bert_weight上
        for (int i = 0; i < num_layer_; i++) {
            this->get_tensor(context, 3 + i, &bert_weights.bert_layer_weights[i].attention_weights.query_weight.kernel);
            this->get_tensor(
                context, 3 + num_layer_ + i, &bert_weights.bert_layer_weights[i].attention_weights.query_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 2 + i,
                             &bert_weights.bert_layer_weights[i].attention_weights.key_weight.kernel);
            this->get_tensor(
                context, 3 + num_layer_ * 3 + i, &bert_weights.bert_layer_weights[i].attention_weights.key_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 4 + i,
                             &bert_weights.bert_layer_weights[i].attention_weights.value_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 5 + i,
                             &bert_weights.bert_layer_weights[i].attention_weights.value_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 6 + i,
                             &bert_weights.bert_layer_weights[i].attention_weights.attention_output_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 7 + i,
                             &bert_weights.bert_layer_weights[i].attention_weights.attention_output_weight.bias);

            this->get_tensor(
                context, 3 + num_layer_ * 8 + i, &bert_weights.bert_layer_weights[i].attn_layernorm_weights.beta);
            this->get_tensor(
                context, 3 + num_layer_ * 9 + i, &bert_weights.bert_layer_weights[i].attn_layernorm_weights.gamma);
            this->get_tensor(context,
                             3 + num_layer_ * 10 + i,
                             &bert_weights.bert_layer_weights[i].ffn_weights.intermediate_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 11 + i,
                             &bert_weights.bert_layer_weights[i].ffn_weights.intermediate_weight.bias);
            this->get_tensor(
                context, 3 + num_layer_ * 12 + i, &bert_weights.bert_layer_weights[i].ffn_weights.output_weight.kernel);
            this->get_tensor(
                context, 3 + num_layer_ * 13 + i, &bert_weights.bert_layer_weights[i].ffn_weights.output_weight.bias);
            this->get_tensor(
                context, 3 + num_layer_ * 14 + i, &bert_weights.bert_layer_weights[i].ffn_layernorm_weights.beta);
            this->get_tensor(
                context, 3 + num_layer_ * 15 + i, &bert_weights.bert_layer_weights[i].ffn_layernorm_weights.gamma);
        }
        bert_weights.post_transformer_layernorm_weights.gamma = nullptr;
        bert_weights.post_transformer_layernorm_weights.beta  = nullptr;

        // 获取注意力类型(融合MHA，非融合MHA等)
        ft::AttentionType attention_type =
            ft::getAttentionType<DataType>(size_per_head_, sm_, remove_padding_, from_seq_len_);

        attention_type = ft::AttentionType::FUSED_MHA;

        if (attention_type == ft::AttentionType::FUSED_MHA) {
            std::cout << "FUSED_MHA" << std::endl;
        }
        else if (attention_type == ft::AttentionType::UNFUSED_MHA) {
            std::cout << "UNFUSED_MHA" << std::endl;
        }
        else if (attention_type == ft::AttentionType::UNFUSED_PADDED_MHA){
            std::cout << "UNFUSED_PADDED_MHA" << std::endl;
        }
        else if (attention_type == ft::AttentionType::FUSED_PADDED_MHA){
            std::cout << "FUSED_PADDED_MHA" << std::endl;
        }

        // 初始化bert kernel
        ft::Bert<DataType> bert = ft::Bert<DataType>(batch_size_,
                                                     from_seq_len_,
                                                     head_num_,
                                                     size_per_head_,
                                                     inter_size_,
                                                     num_layer_,
                                                     sm_,
                                                     q_scaling_,
                                                     stream,
                                                     &cublas_wrapper,
                                                     &allocator,
                                                     true,
                                                     attention_type,
                                                     false,
                                                     ft::ActivationType::Gelu,
                                                     ft::LayerNormType::post_layernorm);

        tf::Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, context->input(0).shape(), &output));
        DataType* out_tensor = reinterpret_cast<DataType*>(output->flat<T>().data());

        ft::DataType            data_type     = ft::getTensorType<DataType>();
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{this->convert_tensor(context->input(0)),
                                                                        this->convert_int_tensor(context->input(2))};

        // 输出
        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
            // 设备类型
            // 数据类型 (float, half, bfloat16)
            // 输出shape
            // 输出具体值（指针, 存储的位置，后续按照shape进行解码）
            ft::Tensor{ft::MEMORY_GPU, 
                       data_type,      
                       std::vector<size_t>{batch_size_, from_seq_len_, (size_t)(head_num_ * size_per_head_)},
                       out_tensor}};

        try {
            // 调用bert的推理
            bert.forward(&output_tensors, &input_tensors, &bert_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << tf::errors::Internal(error.what());
            ft::FT_CHECK(false);
        }
        catch (...) {
            std::cout << tf::errors::Internal("Runtime error");
            ft::FT_CHECK(false);
        }
    }

private:
    int                                head_num_ = 0, size_per_head_ = 0, inter_size_ = 0, num_layer_ = 0;
    float                              q_scaling_ = 1.0f;
    bool                               remove_padding_;
    int                                sm_;
    ft::cublasAlgoMap*                 cublas_algo_map_;
    typedef TFTraits<T>                traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("Bert").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), BertOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
