# Generative remote sensing

## Datasets

```bash
unzip x -d ../decomp
unzip NASC_TG2.zip '*RGB/*' -d ../decomp/NASC_TG2
```

## Models

git clone https://github.com/autonomousvision/stylegan-xl.git
git clone https://ghproxy.com/https://github.com/autonomousvision/stylegan-xl.git

git clone https://github.com/CompVis/taming-transformers.git
git clone https://ghproxy.com/https://github.com/CompVis/taming-transformers.git

curl -o 2021-04-03T19-39-50_cin_transformer.zip "https://app.koofr.net/content/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e/files/get/2021-04-03T19-39-50_cin_transformer.zip?path=%2F&force"




python main.py --base configs/custom_cond_AID_test0.2.yaml -t True --resume logs/2023-05-08T22-57-48_custom_cond_AID_test0.2

python main.py --base configs/custom_vqgan_AID_test0.2.yaml -t True 

42, 1126, 2000, 3407, 31415


ln -s /data/cyc/2023-generative-remote-sensing/datasets /data/cyc/2023-generative-remote-sensing/generative_models/taming-transformers/


ln -s /root/autodl-tmp/2023-generative-remote-sensing/datasets /root/autodl-tmp/2023-generative-remote-sensing/generative_models/taming-transformers/



https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
conda config --set custom_channels.nvidia https://mirrors.cernet.edu.cn/anaconda-extra/cloud/

ImportError: libSM.so.6: cannot open shared object file: No such file or directory
apt-get update
apt-get install libsm6



docker run --rm -d -v wandb:/vol -p 8087:8080  --name wandb-local wandb/local




-------------------------------------------



python dataset_tool.py --source=./data/few-shot-images/pokemon --dest=./data/pokemon256.zip --resolution=256x256 --transform=center-crop



ln -s /data/cyc/2023-generative-remote-sensing/datasets /data/cyc/2023-generative-remote-sensing/generative_models/stylegan-xl/

python train.py --outdir=./training-runs/pokemon --cfg=stylegan3-t --data=./data/AID_test0.2_train.zip \
    --gpus=1 --batch=64 --mirror=1 --snap 10 --batch-gpu 8 --kimg 10000 --syn_layers 10



ln -s /data/cyc/2023-generative-remote-sensing/datasets /data/cyc/2023-generative-remote-sensing/generative_models/guided-diffusion/


MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma True" 
DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule cosine  --rescale_learned_sigmas False --rescale_timesteps False" 
TRAIN_FLAGS="--lr 5e-5 --batch_size 4 --save_interval 10000 --resume_checkpoint logs/256x256_diffusion.pt" 
python scripts/image_train.py --data_flist datasets/data/cls/AID_test0.2/train.flist $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS