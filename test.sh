export CUDA_VISIBLE_DEVICES=0,2
export OMP_NUM_THREADS=64
export NUM_GPUS=2

# 只有--test-only 参数表示只进行测试，他会根据custom_dataset里val_dataloader进行测试，没有test_dataloader选项，需要手动切换
# 加上 --eval-wda 参数可以评估 WDA 指标，会在18个子域上评估

# goldyolo
torchrun --master_port=9931 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/dfine.yml -r outputs/dfine-n-goldyolo/best_stg1.pth --test-only --eval-wda

# # goldyolo-improve
# torchrun --master_port=9932 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/dfine.yml -r outputs/dfine-n-goldyolo-improve/best_stg1.pth --test-only --eval-wda

# # metaformer-assa
# torchrun --master_port=9933 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/dfine.yml -r outputs/dfine-n-metaformer-assa/best_stg1.pth --test-only --eval-wda




# torchrun --master_port=9930 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/deim_hgnetv2_n_custom.yml -r outputs/deim_hgnetv2_n_custom/best_stg2.pth --test-only --eval-wda

# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/deim_r18vd_120e_coco.yml -r outputs/deim_r18vd_120e_custom/best_stg2.pth --test-only --eval-wda
# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/dfine_hgnetv2_n_custom.yml -r outputs/dfine_hgnetv2_n_custom/best_stg2.pth --test-only
# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/dfine_hgnetv2_n_mal_custom.yml -r outputs/dfine_hgnetv2_n_mal_custom/best_stg2.pth --test-only
# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/rtdetrv2_r18vd_120e_coco.yml -r 