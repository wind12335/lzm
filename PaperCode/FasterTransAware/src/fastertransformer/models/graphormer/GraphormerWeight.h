#pragma once

#include "src/fastertransformer/models/graphormer/GraphormerLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct GraphormerWeight {

    GraphormerWeight() = default;
    ~GraphormerWeight();
    std::vector<GraphormerLayerWeight<T>> graphormer_layer_weights;
    LayerNormWeight<T>              post_transformer_layernorm_weights;

    bool isValidLayerParallelId(int l);
    void loadModel(std::string dir_path);

private:
    void setWeightPtr();

    size_t hidden_units_;
    size_t inter_size_;
    size_t num_layer_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    bool   is_maintain_buffer = false;
    T*     weights_ptr[2];
};

}  // namespace fastertransformer