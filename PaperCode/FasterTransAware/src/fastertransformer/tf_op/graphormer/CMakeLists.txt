add_library(tf_graphormer SHARED GraphormerOp.cc)
target_link_libraries(tf_graphormer PRIVATE Graphormer ${tf_link} -lcublas -lcublasLt -lcudart cublasAlgoMap)