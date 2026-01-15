
# 训练
单块GPU
```
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/yaml/deim_dfine_hgnetv2_n_mg.yml
```
多块GPU，2指的是用的块数，和CUDA_VISIBLE_DEVICES=3,5要对应
```
CUDA_VISIBLE_DEVICES=3,5 python -m torch.distributed.run --nproc_per_node 2 train.py -c configs/yaml/deim_dfine_hgnetv2_n_mg.yml
```
CUDA_VISIBLE_DEVICES=1,3,5 可见的GPU，虽然有六块，但是设置之后，只有3，5可见，且内部重新编号为0，1,2
 python -m torch.distributed.run --nproc_per_node 2 多GPU分布式启动代码，使用两块GPU，从低到高开始0,1
 train.py -c configs/deim/deim_hgnetv2_n_custom.yml 训练脚本 -c 是config配置文件，

 大体这个意思，别听他的
 重要的是模块。

 一切的一切要保证接口一致。去找模块添加的位置。DEIM-DEIM/engine/extre_module/custom_nn。这是一个付费的项目

 cfg-improve 是他弄好的一些现成的模型了，是可以直接跑的一般。

 行，目前为止有问题吗。没问题就好，继续。

# 测试
```
python train.py -c configs/test/dfine_hgnetv2_n_visdrone.yml --test-only -r /home/waas/best_stg2.pth
```


# 模型说明
- base/ : 基础配置文件
    - dataloader_dfine.yml ： dfine的默认数据预处理相关
    - dataloader_rtdetrv2.yml
    - dataloader.yml
    - deimv2.yml
    - deim.yml
    - dfine_hgnetv2.yml
    - optimizer.yml
    - rt_deim.yml
    - rtdetrv2_r50vd.yml
    - rt_optimizer.yml
- baseline/
- cfg/ : 类似YOLO的模型配置文件，只包含模型，这是DFINE的原始方法，多个尺度变体
- cfg-improve/ : 这是经过创新的模型配置，纯模型
- dataset/
- deim/
- deim_rtdetrv2/
- deimv2/
- dfine/
- dinov3/
- distill/
- runtime.yml
- test/
- yaml/

# 训练配置
在训练时，需要接收一个包含完整实验配置的配置文件(.yml)，它包含：
1. 数据
2. 模型
3. 评估器
4. 优化器
5. 过程中配置

他们经过继承和重写，集成到一个yml中。
一般，位于yaml目录。