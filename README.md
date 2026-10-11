# 数据集
1.计算图结构的数据集分别使用NasNet、PnasNet、InceptionV3、GNMT、NMT，用于图编码模型准确性预测的数据集分别为pcqm4m和ZINC数据集，数据集地址均为PaperCode/TransAware/transaware/datasets。
2.计算图结构的数据集保存方式为pkl文件，用pickle.load直接可以加载。图编码准确性预测的分子数据集使用torch_geometric.datasets加载。

# 运行方法
本实验提供脚本均可通过bash+脚本的形式执行，可通过重定向获取各个部分的结果。
1.TransAware：
实验1：双层节点融合机制的运行脚本在PaperCode/TransAware/transaware/shell/merge下。
实验2：图编码模型在不同阈值融合下执行时间的脚本在PaperCode/TransAware/transaware/shell/graphormer_time下。
实验3：TransAware策略搜索的脚本在PaperCode/TransAware/transaware/shell/strategy_execution_time下。
实验4：图编码准确性分析的文件分别为PaperCode/TransAware/transaware/下的test_PCQM4M.py和test_ZINC.py，可直接python执行。
2.FastTransAware
实验1：策略搜索执行时间的脚本为PaperCode/FasterTransAware/build/fastertransaware.sh。
实验2：FastGraphormer执行的脚本为/home/PaperCode/FasterTransAware/build/graphormer_time_model.sh和/home/PaperCode/FasterTransAware/build/graphormer_time.sh。
实验3：FastLSTM执行的脚本与实验1相同，在代码中插入time即可获取FastLSTM执行时间。
实验4：内存开销的执行脚本与实验1相同，本文通过nvitop监控GPU显存变化率，获取FastTransAware各个模块的内存开销。
实验5：FastGEMM和FastSoftmax的执行时间均可由实验1中的脚本打印。
3.baseline
baseline的实验脚本分别在PaperCode/TransAware/Hierarchical/Trinity/baseline_gnmt_aware.sh以及/home/PaperCode/TransAware/Hierarchical/Trinity/hierarchical_test.sh中。

# 代码注解
```text
├── PaperCode
    ├── FasterTransAware                 // 论文第二个方法
    │   ├── build                        // 核函数分块和融合的c++编译部分，包含实验执行脚本
    │   ├── examples                     // 包含了fasttransaware
    │   │   ├── tensorflow               // 包含了fasttransaware
    │   │   |   ├── graphormer           // fasttransaware的主体部分
    │   ├── src                          // fasttransaware的c++实现部分
    ├── TransAware                       // 论文第一个方法   
    │   ├── Hierarchical                 // 实验基线，可用于TransAware和FastTransAware的对比
    │   │   ├── Trinity_main_test.py     // 基线执行入口
    │   │   ├── graph_scheduling.py      // aware基线主体
    │   │   ├── graph_scheduling_base.py // Hierarchical基线主体
    │   ├── transaware                   // 论文第一个方法
    │   │   ├── datasets                 // 数据集
    │   │   ├── graph.py                 // 策略搜索所需的图结构   
    │   │   ├── embedding_network.py     // Graphormer模型文件
    │   │   ├── strategy_search.py       // transaware入口
    │   │   ├── LSTM.py                  // LSTM策略搜索模块
    │   │   ├── merge.py                 // 双层节点融合机制文件
    │   │   ├── simulator.py             // 模拟执行引擎文件
    │   │   ├── progressive_placer.py    // 并行策略搜索和放置主体部分 
```

# 结果
1.TransAware：
实验1双层节点融合机制，表5.3的结果在merge.py第90行，表5.4的结果在progressive_placer.py第123行。
实验2策略性能分析表5.5的结果在progressive_placer.py第174行。
实验3图5.1和图5.2的结果分别在test_ZINC.py第126行和test_PCQM4M.py第128行。
2.FastTransAware
实验1图5.3和图5.4的结果在PaperCode/FasterTransAware/examples/tensorflow/graphormer/graphormer_example.py第129和130行。
实验2图5.5的结果在PaperCode/FasterTransAware/examples/tensorflow/graphormer/progressive_placer.py第191行。
实验3图5.6的结果通过开启两个终端，在一个终端执行时，另一个终端开启nvitop指令监控内存变化，计算内存开销。
实验4图5.7的结果在PaperCode/FasterTransformer/3rdparty/trt_fused_multihead_attention/qkvToContext.cu第689行。
实验4图5.8的结果在PaperCode/FasterTransformer/src/fastertransformer/utils/gemm_test/encoder_gemm_func.cc第236行。
