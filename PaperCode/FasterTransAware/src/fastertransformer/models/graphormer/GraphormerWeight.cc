/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "GraphormerWeight.h"

namespace fastertransformer {

template<typename T>
GraphormerWeight<T>::~GraphormerWeight()
{
    if (is_maintain_buffer == true) {
        graphormer_layer_weights.clear();
        for (int i = 0; i < 2; i++) {
            deviceFree(weights_ptr[i]);
        }

        post_transformer_layernorm_weights.gamma = nullptr;
        post_transformer_layernorm_weights.beta  = nullptr;
        is_maintain_buffer                       = false;
    }
}

template<typename T>
void GraphormerWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "bert");
    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            graphormer_layer_weights[l].loadModel(dir_path + "model.encoder.layer." + std::to_string(l) + ".",
                                                  model_file_type);
        }
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__, " stop");
}

template<typename T>
bool GraphormerWeight<T>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_rank_)
           && (l < local_num_layer * (pipeline_para_rank_ + 1));
}

template<typename T>
void GraphormerWeight<T>::setWeightPtr()
{
    post_transformer_layernorm_weights.gamma = weights_ptr[0];
    post_transformer_layernorm_weights.beta  = weights_ptr[1];

    is_maintain_buffer = true;
}

template struct GraphormerWeight<float>;
template struct GraphormerWeight<half>;
#ifdef ENABLE_BF16
template struct GraphormerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
