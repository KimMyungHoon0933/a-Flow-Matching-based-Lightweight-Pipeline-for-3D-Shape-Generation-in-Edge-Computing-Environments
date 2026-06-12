# Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models
[In Cho](https://sites.google.com/view/im-join), [Youngbeom Yoo](https://sites.google.com/view/youngbeomyoo/home), [Subin Jeon](https://sites.google.com/yonsei.ac.kr/subinjeon/home), [Seon Joo Kim](https://sites.google.com/site/seonjookim/)

This is the official implementation of the paper "[Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models](https://arxiv.org/abs/2503.08737)" (ICCV 2025).

## Installation
### Requirements
We recommend using a virtual environment to avoid conflicts with other packages.
```bash
conda create -n cod python=3.9
conda activate cod
pip install -r requirements.txt
```
The [pointops](https://github.com/Silverster98/pointops) package should also be installed, of which source code is attached in the external directory:
```bash
pip install ./external/pointops
```

Our code is tested on Python 3.9, torch 2.1.0, and CUDA 12.2.

We also provide prebuilt [docker image](https://hub.docker.com/r/join16/cod-vae), which contains all the dependencies required to run the code.
```bash
docker pull join16/cod-vae:cu122-py310
```
The same environment can be built by running the following command:
```bash
docker build -t cod-vae:cu122-py310 .
```

## Dataset
### ShapeNet
We use the preprocessed version of the ShapeNet dataset, provided in [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet).
To enhance the IO utilization, we further process the provided data into HDF5 format.
First, download the above dataset, and make sure to have the following directory structure:
```
{raw_root_dir}/
    ├── ShapeNetV2_point/
    └── ShapeNetV2_surface/
```
Then, change `raw_root_dir` to the path of the downloaded data in `config/data/shapenet.yaml`.
Finally, set the `root_dir` to the destination directory, run the following command to create HDF5 files:
```bash
python preprocess_dataset.py --data shapenet
```

## Pretrained weights
Pretrained weights can be downloaded from [here](https://drive.google.com/drive/folders/1aJE_LbnyV8lBqjRc7tcXjui52N1Kvvjm).
Currently, we provide the weights of VAE trained on Shapenet.

## Training
### Autoencoder (stage 1)
To train the autoencoder, run the following command:
```bash
python train.py config/train_stage1.yaml -n {experiment_name} -g {gpu_ids}
```
Please run `python train.py -h` to see the available command line arguments.

### VAE (stage 2)
Once the autoencoder is trained, modify the `autoencoder_checkpoint_path` in `config/train_stage2.yaml` to the path of the trained autoencoder weights.
Then, run the following command to train the VAE:
```bash
python train.py config/train_stage2.yaml
```

## Inference
To test the trained autoencoders, run the following command:
```bash
python evaluate_ae.py {weights_dir} --data shapenet
```
`{weights_dir}` can be either the directory of the pretrained weights (`weights/vae_m32`) or the directory of your trained outputs (`logs/your_experiments`).


## Acknowledgements
Our code is built upon the [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) and the codebase of [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet).
We sincerely appreciate the authors for sharing their code and data, which greatly helped our research.

## <a name="citation"></a> Citation
```BibTex
@inproceedings{cho2025cod,
  author={Cho, In and Yoo, Youngbeom and Jeon, Subin and Kim, Seon Joo},
  title={Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```