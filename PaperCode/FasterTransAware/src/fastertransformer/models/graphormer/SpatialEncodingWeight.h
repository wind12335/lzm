#pragma once

#include "src/fastertransformer/layers/DenseWeight.h"

namespace fastertransformer {

template<typename T1, typename T2 = T1>
struct SpatialEncodingWeight {
    DenseWeight<T1, T2> dense1_weight;
    DenseWeight<T1, T2> dense2_weight;
};

}  // namespace fastertransformer
