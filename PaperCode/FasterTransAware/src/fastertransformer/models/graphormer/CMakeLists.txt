cmake_minimum_required(VERSION 3.8)

add_library(GraphormerLayerWeight STATIC GraphormerLayerWeight.cc)
set_property(TARGET GraphormerLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GraphormerLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GraphormerLayerWeight PUBLIC memory_utils)

add_library(GraphormerWeight STATIC GraphormerWeight.cc)
set_property(TARGET GraphormerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET GraphormerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(GraphormerWeight PUBLIC GraphormerLayerWeight)

add_library(Graphormer STATIC Graphormer.cc)
set_property(TARGET Graphormer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Graphormer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Graphormer PUBLIC -lcudart bert_preprocess_kernels cublasMMWrapper 
                      UnfusedAttentionLayer FusedAttentionLayer TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                      layernorm_kernels add_residual_kernels GraphormerWeight nccl_utils custom_ar_comm tensor cuda_utils logger)

# add_executable(bert_gemm bert_gemm.cc)
# target_link_libraries(bert_gemm PUBLIC -lcublas -lcublasLt -lcudart encoder_gemm_func encoder_igemm_func memory_utils tensor cuda_utils logger)