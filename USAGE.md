
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
