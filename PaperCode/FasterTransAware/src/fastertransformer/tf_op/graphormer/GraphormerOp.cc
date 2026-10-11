#include "src/fastertransformer/models/graphormer/Graphormer.h"
#include "src/fastertransformer/tf_op/BaseOp.h"

namespace ft = fastertransformer;
namespace tf = tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Graphormer")
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
    .Input("attr_layernorm_beta: N * T")
    .Input("attr_layernorm_gamma: N * T")
    .Input("inter_kernel: N * T")
    .Input("inter_bias: N * T")
    .Input("output_kernel: N * T")
    .Input("output_bias: N * T")
    .Input("ffn_layernorm_beta: N * T")
    .Input("ffn_layernorm_gamma: N * T")
    .Input("attr_spatial_encoding_dense1_kernel: N * T")
    .Input("attr_spatial_encoding_dense1_bias: N * T")
    .Input("attr_spatial_encoding_dense2_kernel: N * T")
    .Input("attr_spatial_encoding_dense2_bias: N * T")
    .Output("output: T")
    .Attr("N: int")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("inter_size: int >= 1")
    .Attr("num_layer: int >= 1")
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
class GraphormerOp: public BaseOp<T> {
public:
    explicit GraphormerOp(tf::OpKernelConstruction* context): BaseOp<T>(context)
    {   
        try {
            // 初始化参数并赋值
            OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
            OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
            OP_REQUIRES_OK(context, context->GetAttr("inter_size", &inter_size_));
            OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
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

    ~GraphormerOp()
    {
        delete cublas_algo_map_;
    }

    void Compute(tf::OpKernelContext* context) override
    {   

        // REGISTER_OP里面input的数量, BERT是3(input, output, seq_len) + 12个layers * 16个参数每层layer
        OP_REQUIRES(context,
                    context->num_inputs() == (num_layer_ * 20) + 3,
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

        // 存储graphormer的权重
        ft::GraphormerWeight<DataType> graphormer_weights;
        graphormer_weights.graphormer_layer_weights.resize(num_layer_);

        // 将输入的权重全部转移到bert_weight上
        for (int i = 0; i < num_layer_; i++) {
            this->get_tensor(context, 3 + i, &graphormer_weights.graphormer_layer_weights[i].attention_weights.query_weight.kernel);
            this->get_tensor(
                context, 3 + num_layer_ + i, &graphormer_weights.graphormer_layer_weights[i].attention_weights.query_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 2 + i,
                             &graphormer_weights.graphormer_layer_weights[i].attention_weights.key_weight.kernel);
            this->get_tensor(
                context, 3 + num_layer_ * 3 + i, &graphormer_weights.graphormer_layer_weights[i].attention_weights.key_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 4 + i,
                             &graphormer_weights.graphormer_layer_weights[i].attention_weights.value_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 5 + i,
                             &graphormer_weights.graphormer_layer_weights[i].attention_weights.value_weight.bias);
            this->get_tensor(context,
                             3 + num_layer_ * 6 + i,
                             &graphormer_weights.graphormer_layer_weights[i].attention_weights.attention_output_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 7 + i,
                             &graphormer_weights.graphormer_layer_weights[i].attention_weights.attention_output_weight.bias);

            this->get_tensor(
                context, 3 + num_layer_ * 8 + i, &graphormer_weights.graphormer_layer_weights[i].attn_layernorm_weights.beta);
            this->get_tensor(
                context, 3 + num_layer_ * 9 + i, &graphormer_weights.graphormer_layer_weights[i].attn_layernorm_weights.gamma);
            
            this->get_tensor(context,
                             3 + num_layer_ * 10 + i,
                             &graphormer_weights.graphormer_layer_weights[i].ffn_weights.intermediate_weight.kernel);
            this->get_tensor(context,
                             3 + num_layer_ * 11 + i,
                             &graphormer_weights.graphormer_layer_weights[i].ffn_weights.intermediate_weight.bias);
            this->get_tensor(
                context, 3 + num_layer_ * 12 + i, &graphormer_weights.graphormer_layer_weights[i].ffn_weights.output_weight.kernel);
            this->get_tensor(
                context, 3 + num_layer_ * 13 + i, &graphormer_weights.graphormer_layer_weights[i].ffn_weights.output_weight.bias);

            this->get_tensor(
                context, 3 + num_layer_ * 14 + i, &graphormer_weights.graphormer_layer_weights[i].ffn_layernorm_weights.beta);
            this->get_tensor(
                context, 3 + num_layer_ * 15 + i, &graphormer_weights.graphormer_layer_weights[i].ffn_layernorm_weights.gamma);

            this->get_tensor(
                context, 3 + num_layer_ * 16 + i, &graphormer_weights.graphormer_layer_weights[i].spatial_encoding_weights.dense1_weight.kernel);
            this->get_tensor(
                context, 3 + num_layer_ * 17 + i, &graphormer_weights.graphormer_layer_weights[i].spatial_encoding_weights.dense1_weight.bias);
            this->get_tensor(
                context, 3 + num_layer_ * 18 + i, &graphormer_weights.graphormer_layer_weights[i].spatial_encoding_weights.dense2_weight.kernel);
            this->get_tensor(
                context, 3 + num_layer_ * 19 + i, &graphormer_weights.graphormer_layer_weights[i].spatial_encoding_weights.dense2_weight.bias);
        }
        graphormer_weights.post_transformer_layernorm_weights.gamma = nullptr;
        graphormer_weights.post_transformer_layernorm_weights.beta  = nullptr;

        // 获取注意力类型(融合MHA，非融合MHA等)
        ft::AttentionType attention_type = ft::AttentionType::FUSED_MHA;
        // 初始化graphormer kernel
        ft::Graphormer<DataType> graphormer = ft::Graphormer<DataType>(
                                                batch_size_,
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
            graphormer.forward(&output_tensors, &input_tensors, &graphormer_weights);
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
    int                                sm_;
    ft::cublasAlgoMap*                 cublas_algo_map_;
    typedef TFTraits<T>                traits_;
    typedef typename traits_::DataType DataType;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                                                \
    REGISTER_KERNEL_BUILDER(Name("Graphormer").Device(tf::DEVICE_GPU).TypeConstraint<T>("T"), GraphormerOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
