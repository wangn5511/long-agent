# 使用mobilenet v4的docker
docker exec -it mobilenetv4_container_ymy /bin/bash

docker start mobilenetv4_container_ymy

docker stop mobilenetv4_container_ymy
docker rm mobilenetv4_container_ymy

conda activate mobilenetv4

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com 

# cd /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master

----------------------------------------------------------------------------------------
# todo 注意这里用merged 和test  后续改   /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/utils/dataloaders.py 225
export CUDA_VISIBLE_DEVICES=1
python imagenet.py \
  -a mobilenetv2 \
  -d /home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_new_split \
  -b 128 \
  --epochs 200 \
  --save-step 10 \
  --num-classes 8 \
  --lr-decay cos \
  --lr 0.05 \
  --wd 4e-5 \
  --width-mult 1.0 \
  --input-size 224 \
  -j 4 \
  --checkpoint /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_truth \
  --resume /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2/mobilenetv2_epoch100_acc99.84_checkpoint.pth


----------------------------------------------------------------------------

cd /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master

# 导出 mobilenetv2，类别数 8，batch_size=1，输入 224x224
python onnx_export_v2.py \
  --model mobilenetv2 \
  --dynamic-batch \
  --checkpoint ./checkpoints_mbv2_truth/model_best.pth.tar \
  --nb-classes 8 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_truth/mobilenetv2_truth_best_dynamic_batch.onnx

----------------------------------------------------------------------------
# num-classes 10
export CUDA_VISIBLE_DEVICES=1
python imagenet.py \
  -a mobilenetv2 \
  -d /home/models/hagrid_labeled_add_10 \
  -b 128 \
  -j 4 \
  --epochs 100 \
  --save-step 10 \
  --num-classes 10 \
  --lr-decay cos \
  --lr 0.05 \
  --wd 4e-5 \
  --width-mult 1.0 \
  --input-size 224 \
  --checkpoint /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10 \
  --pretrained /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/pretrained/mobilenetv2_1.0-0c6065bc.pth


# 导出 mobilenetv2，类别数 10，输入 224x224
python onnx_export_v2.py \
  --model mobilenetv2 \
  --dynamic-batch \
  --checkpoint ./checkpoints_mbv2_10/model_best.pth.tar \
  --nb-classes 10 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_10/mobilenetv2_10_best_dynamic_batch.onnx

----------------------------------------------------------------------------

# num-classes 10  真实数据finetune
export CUDA_VISIBLE_DEVICES=1
python imagenet.py \
  -a mobilenetv2 \
  -d /home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_truth_new_all_split \
  -b 128 \
  -j 4 \
  --epochs 100 \
  --save-step 10 \
  --num-classes 10 \
  --lr-decay cos \
  --lr 0.05 \
  --wd 4e-5 \
  --width-mult 1.0 \
  --input-size 224 \
  --checkpoint /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10_truth \
  --pretrained /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10/mobilenetv2_epoch100_acc99.84_checkpoint.pth


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --dynamic-batch \
  --checkpoint ./checkpoints_mbv2_10_truth/model_best.pth.tar \
  --nb-classes 10 \
  --batch-size 1 \
  --img-size 224 \
  --dynamic-batch \
  --output ./checkpoints_mbv2_10_truth/mobilenetv2_10_truth_best.onnx


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --dynamic-batch \
  --checkpoint ./checkpoints_mbv2_10_truth/model_best.pth.tar \
  --nb-classes 10 \
  --img-size 224 \
  --output ./checkpoints_mbv2_10_truth/mobilenetv2_10_truth_best.onnx


python onnx_validate_all_acc.py \
 --data_root=/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_truth_new_all_split/val \
 --onnx-input=/home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10_truth/mobilenetv2_10_truth_best_dynamic_batch.onnx  \
 --batch-size=1 \
 --img-size=224 

-----------------------------------------------------------------------------------
# num-classes 8   新的数据
export CUDA_VISIBLE_DEVICES=1
python imagenet.py \
  -a mobilenetv2 \
  -d /home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_truth_new_all_split_8 \
  -b 128 \
  -j 4 \
  --epochs 100 \
  --save-step 10 \
  --num-classes 8 \
  --lr-decay cos \
  --lr 0.05 \
  --wd 4e-5 \
  --width-mult 1.0 \
  --input-size 224 \
  --checkpoint /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_truth_new_new \
  --pretrained /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2/model_best.pth.tar


python onnx_export_v2.py \
  --model mobilenetv2 \
  --checkpoint ./checkpoints_mbv2_truth_new_new/model_best.pth.tar \
  --nb-classes 8 \
  --batch-size 1 \
  --img-size 224 \
  --dynamic-batch \
  --output ./checkpoints_mbv2_truth_new_new/mobilenetv2_8_truth_best_dynamic_new.onnx


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --checkpoint ./checkpoints_mbv2_truth_new_new/model_best.pth.tar \
  --nb-classes 8 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_truth_new_new/mobilenetv2_8_truth_best_new.onnx


python onnx_validate_all_acc.py \
 --data_root=/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_truth_new_all_split_8/val \
 --onnx-input=/home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_truth_new/mobilenetv2_8_truth_best_dynamic_new.onnx \
 --batch-size=1 \
 --img-size=224 

---------------------------------------------------------------------
# num-classes 10   新的数据  finetune
export CUDA_VISIBLE_DEVICES=1
python imagenet.py \
  -a mobilenetv2 \
  -d /home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split_random_2 \
  -b 128 \
  -j 4 \
  --epochs 100 \
  --save-step 10 \
  --num-classes 10 \
  --lr-decay cos \
  --lr 0.05 \
  --wd 4e-5 \
  --width-mult 1.0 \
  --input-size 224 \
  --checkpoint /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10_truth_new_random_2 \
  --pretrained /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10/mobilenetv2_epoch100_acc99.84_checkpoint.pth


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --dynamic-batch \
  --checkpoint ./checkpoints_mbv2_10_truth_new_random/model_best.pth.tar \
  --nb-classes 10 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_10_truth_new_random/mobilenetv2_10_truth_best_dynamic.onnx


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --checkpoint ./checkpoints_mbv2_10_truth_new_random/model_best.pth.tar \
  --nb-classes 10 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_10_truth_new_random/mobilenetv2_10_truth_best.onnx


python onnx_validate_all_acc.py \
 --data_root=/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split/val \
 --onnx-input=/home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10_truth_new_random/mobilenetv2_10_truth_best_dynamic.onnx \
 --batch-size=1 \
 --img-size=224 


-----------------------------------------------------------------------------------
# num-classes 10   新的数据  finetune add Gun data
export CUDA_VISIBLE_DEVICES=1
python imagenet.py \
  -a mobilenetv2 \
  -d /home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split_random_2 \
  -b 128 \
  -j 4 \
  --epochs 100 \
  --save-step 10 \
  --num-classes 10 \
  --lr-decay cos \
  --lr 0.05 \
  --wd 4e-5 \
  --width-mult 1.0 \
  --input-size 224 \
  --checkpoint /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10_truth_new_random_2 \
  --pretrained /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10/mobilenetv2_epoch100_acc99.84_checkpoint.pth


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --dynamic-batch \
  --checkpoint ./checkpoints_mbv2_10_truth_new_random_2/model_best.pth.tar \
  --nb-classes 10 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_10_truth_new_random_2/mobilenetv2_10_truth_best_dynamic.onnx


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --checkpoint ./checkpoints_mbv2_10_truth_new_random_2/model_best.pth.tar \
  --nb-classes 10 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_10_truth_new_random_2/mobilenetv2_10_truth_best.onnx


python onnx_validate_all_acc.py \
 --data_root=/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split_random_2/test \
 --onnx-input=/home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10_truth_new_random_2/mobilenetv2_10_truth_best_dynamic.onnx \
 --batch-size=1 \
 --img-size=224 

-----------------------------------------------------------------------------------
# num-classes 10   新的数据  finetune add Gun data 和FIST data
export CUDA_VISIBLE_DEVICES=1
python imagenet.py \
  -a mobilenetv2 \
  -d /home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split_random_2 \
  -b 128 \
  -j 4 \
  --epochs 100 \
  --save-step 10 \
  --num-classes 10 \
  --lr-decay cos \
  --lr 0.05 \
  --wd 4e-5 \
  --width-mult 1.0 \
  --input-size 224 \
  --checkpoint /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10_truth_new_random_2_add_fist_gun \
  --pretrained /home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10/mobilenetv2_epoch100_acc99.84_checkpoint.pth


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --dynamic-batch \
  --checkpoint ./checkpoints_mbv2_10_truth_new_random_2_add_fist_gun/model_best.pth.tar \
  --nb-classes 10 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_10_truth_new_random_2_add_fist_gun/mobilenetv2_10_truth_best_dynamic.onnx


  python onnx_export_v2.py \
  --model mobilenetv2 \
  --checkpoint ./checkpoints_mbv2_10_truth_new_random_2_add_fist_gun/model_best.pth.tar \
  --nb-classes 10 \
  --batch-size 1 \
  --img-size 224 \
  --output ./checkpoints_mbv2_10_truth_new_random_2_add_fist_gun/mobilenetv2_10_truth_best.onnx


python onnx_validate_all_acc.py \
 --data_root=/home/yaomingyuan/Program/gesture_recognition/MobileNetV4/image_datasets/recordings_clean0_all_split_random_2/test \
 --onnx-input=/home/yaomingyuan/Program/gesture_recognition/mobilenetv2.pytorch-master/checkpoints_mbv2_10_truth_new_random_2_add_fist_gun/mobilenetv2_10_truth_best_dynamic.onnx \
 --batch-size=1 \
 --img-size=224 