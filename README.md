# Generative remote sensing


## Environment

https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
/hy-tmp/miniconda3
https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
https://developer.aliyun.com/mirror/anaconda
https://mirrors.sustech.edu.cn/help/anaconda.html#configuration
https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

conda config --set custom_channels.nvidia https://mirrors.cernet.edu.cn/anaconda-extra/cloud/
conda clean -i

ln -s /root/autodl-tmp/miniconda3 /root/miniconda3

gpushare-cli baidu down /2023_generative_remote_sensing/pre-trained/stylegan-xl/imagenet256.pkl .

git config --global user.name "yc-cui"
git config --global user.email "2548846280@qq.com"
## Datasets

gpushare-cli baidu down /2023_generative_remote_sensing/datasets/AID/AID_dataset.zip


```bash
unzip x -d ../decomp
unzip NASC_TG2.zip '*RGB/*' -d ../decomp/NASC_TG2
```

## Models


### taming-transformers

gpushare-cli baidu down /2023_generative_remote_sensing/pre-trained/taming-transformers/vqgan_imagenet_f16_16384.ckpt /hy-tmp/2023_generative_remote_sensing/generative_models/taming-transformers/logs/

conda env create -f environment.yaml
conda activate taming

curl -o 2021-04-03T19-39-50_cin_transformer.zip "https://app.koofr.net/content/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e/files/get/2021-04-03T19-39-50_cin_transformer.zip?path=%2F&force"




python main.py --base configs/custom_cond_AID_test0.2.yaml -t True --resume logs/2023-05-08T22-57-48_custom_cond_AID_test0.2

python main.py --base configs/custom_vqgan_AID_test0.2.yaml -t True 
python main.py --base configs/custom_vqgan_AID_test0.2.yaml -t True ;  /usr/bin/shutdown   

python main.py --base configs/custom_transformer_AID_test0.2.yaml -t True --resume logs/2023-05-15T14-17-42_custom_transformer_AID_test0.2;  /usr/bin/shutdown   

42, 1126, 2000, 3407, 31415


ln -s /data/cyc/2023-generative-remote-sensing/datasets /data/cyc/2023-generative-remote-sensing/generative_models/taming-transformers/


ln -s /root/autodl-tmp/2023-generative-remote-sensing/datasets /root/autodl-tmp/2023-generative-remote-sensing/generative_models/taming-transformers/


ln -s /hy-tmp/2023-generative-remote-sensing/datasets /hy-tmp/2023-generative-remote-sensing/generative_models/taming-transformers/

ImportError: libSM.so.6: cannot open shared object file: No such file or directory
apt-get update
apt-get install libsm6



docker run --rm -d -v wandb:/vol -p 8087:8080  --name wandb-local wandb/local

VQGAN ImageNet (f=16), 16384
https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/


python scripts/sample_fast.py -r logs/temp -n 50 -k 600 -t 1.0 -p 0.92 --batch_size 5 --classes 27,28,29


------------------------------------------------------------------------------------


ln -s /data/cyc/2023-generative-remote-sensing/datasets /data/cyc/2023-generative-remote-sensing/generative_models/stylegan-xl/


python dataset_tool.py --source=./datasets/data/cls/AID_test0.2/train.flist --dest=./data/AID_test0.2_train_256.zip --resolution=256x256 --transform=center-crop


python dataset_tool.py --source=./datasets/data/cls/AID_test0.2/train.flist --dest=./data/AID_test0.2_train_128.zip --resolution=128x128 --transform=center-crop

CUDA_VISIBLE_DEVICES=1 python train.py --outdir=./training-runs/AID_test0.2_train_256 --cfg=stylegan3-t --data=./data/AID_test0.2_train_256.zip --gpus=1 --batch=4 --mirror=1 --snap 10 --batch-gpu 4 --kimg 100 --syn_layers 31 --cond True --seed=42 --resume ./logs/imagenet256.pkl

fatal error: cuda_runtime_api.h: No such file or directory
https://github.com/HawkAaron/warp-transducer/issues/15#issuecomment-467668750
locate cuda_runtime_api.h
export CUDA_HOME=/usr/local/cuda-11.4 # change to your path
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"


Downloading: "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth" to /home/dell/.cache/torch/hub/checkpoints/deit_small_distilled_patch16_224-649709d9.pth
wget -O ~/.cache/torch/hub/checkpoints/deit_small_distilled_patch16_224-649709d9.pth "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"


RuntimeError: No such operator aten::cudnn_convolution_backward_weight
https://github.com/autonomousvision/stylegan-xl/issues/23#issuecomment-1101239525
https://discuss.pytorch.org/t/at-has-no-member-cudnn-convolution-xxx/163716/4
https://github.com/nv-tlabs/GET3D/issues/96#issuecomment-1434040703

OK: https://github.com/NVlabs/stylegan3/commit/407db86e6fe432540a22515310188288687858fa

------------------------------------------------------------------------------------


mamba env create -f environment.yml


ln -s /data/cyc/2023-generative-remote-sensing/datasets /data/cyc/2023-generative-remote-sensing/generative_models/guided-diffusion/


ln -s /root/autodl-tmp/2023-generative-remote-sensing/datasets /root/autodl-tmp/2023-generative-remote-sensing/generative_models/guided-diffusion/

bash scripts/image_train.sh configs/custom_train_AID_test0.2.yaml
bash scripts/image_sample.sh  configs/custom_sample_AID_test0.2.yaml


script_util.py NUM_CLASSES = 30
dist_util.py CUDA_VISIBLE_DEVICES
image_datasets.py num_workers=14