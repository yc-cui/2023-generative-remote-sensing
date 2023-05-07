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




python main.py --base configs/custom_cond_AID_test0.2.yaml -t True --gpus 0,


42, 1126, 2000, 3407, 31415


ln -s /data/cyc/2023-generative-remote-sensing/datasets /data/cyc/2023-generative-remote-sensing/generative_models/taming-transformers/