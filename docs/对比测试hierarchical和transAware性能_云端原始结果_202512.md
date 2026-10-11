# 测试hierarchical和transaware

## 环境搭建

1. 创建账号

   扫码注册账号，申请华东三区【乌镇】资源。

   ![a9e0e892314157289eca10970fd2d021](https://pic-store-1390058452.cos.ap-shanghai.myqcloud.com/public/a9e0e892314157289eca10970fd2d021.jpg)

   ![f3be35cca282055103b415023416ca6d](https://pic-store-1390058452.cos.ap-shanghai.myqcloud.com/public/f3be35cca282055103b415023416ca6d.jpg)

2. 上传文件

   将文件上传到账号文件目录

3. 创建容器

   申请成功后，打开控制台，创建容器，创建成功后进入容器。

   ![image-20251208145901587](https://pic-store-1390058452.cos.ap-shanghai.myqcloud.com/public/image-20251208145901587.png)

4. 安装还欠缺的包

   ```shell
   pip install "gensim<4.0.0"
   pip install fastdtw==0.3.4
   pip install numpy==1.18.5  #镜像默认的numpy版本太高了
   ```

5. 配置网络环境

   ```shell
   source /路径/proxy.sh   #先把这个文件上传到账号文件
   ```

# 测试

### 测试hierarchical

1. 进入`PaperCode/TransAware/Hierarchical/Trinity`目录，运行`hierarchical_test.sh`脚本

   ```shell
   # 进入目录
   cd /你的目录/PaperCode/TransAware/Hierarchical/Trinity
   
   # 修改数据集路径
   vi hierarchical_test.sh
   # 修改为需要的数据集的路径（以GNMT为例）
   --pickled-inp-file "./datasets/GNMT/gnmt_baseline.pkl" \
   
   # 执行脚本
   ./hierarchical_test.sh
   ```

2. 结果

   - 模型：NMT_4layers

     ![](https://pic-store-1390058452.cos.ap-shanghai.myqcloud.com/public/A16CE7EB4BB01C243C49FEDC79045844.png)
     
   - 模型：GNMT
   
     ![image-20251208191254131](https://pic-store-1390058452.cos.ap-shanghai.myqcloud.com/public/image-20251208191254131.png)

### 测试transAware

1. 进入`PaperCode/TransAware/transaware`目录，运行`strategy_search.py`

   ```shell
   # 进入目录
   cd /你的目录/PaperCode/TransAware/transaware
   
   # 执行脚本（更改模型的话请改model_name，下面以NMT_4layers为例）
   python strategy_search.py --model_name NMT_4layers --n_devs 4
   ```

2. 结果：

   - 模型：NMT_4layers

     ![](https://pic-store-1390058452.cos.ap-shanghai.myqcloud.com/public/1fef2f22-1329-4f44-8499-ce82d6f3f4c5.png)
     
   - 模型：GNMT
   
     ![image-20251208190433736](https://pic-store-1390058452.cos.ap-shanghai.myqcloud.com/public/image-20251208190433736.png)

## 结果对比

|      模型       | hierarchical | transAware | 性能提升 |
| :-------------: | :----------: | :--------: | :------: |
| **NMT_4layers** |   230.52s    |  139.51s   |  65.2%   |
|    **GNMT**     |   237.98s    |  155.92s   |  52.5%   |
