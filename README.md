# Pytorch implementation of One-Shot Unsupervised Cross Domain Translation ([arxiv](https://arxiv.org/abs/1806.06029)).

Prerequisites
--------------
- Python 3.6
- Pytorch 0.4
- Numpy/Scipy/Pandas
- Progressbar
- OpenCV
- [visdom](https://github.com/facebookresearch/visdom)
- [dominate](https://github.com/Knio/dominate)

## MNIST-to-SVHN and SVHN-to-MNIST

To train autoencoder for both MNIST and SVHN (In mnist_to_svhn folder):
python main_autoencoder.py --use_augmentation=True

To train OST for MNIST to SVHN:
python main_mnist_to_svhn.py --pretrained_g=True --save_models_and_samples=True --use_augmentation=True --one_way_cycle=True --freeze_shared=False

To train OST for SVHN to MNIST:
python main_svhn_to_mnist.py --pretrained_g=True --save_models_and_samples=True --use_augmentation=True --one_way_cycle=True --freeze_shared=False

## Drawing and Style Transfer Tasks

### Download Dataset

To download dataset (in drawing_and_style_transfer folder):
bash datasets/download_cyclegan_dataset.sh $DATASET_NAME
where DATASET_NAME is one of (facades, cityscapes, maps, monet2photo, summer2winter_yosemite)

### Train Autoencoder

To train autoencoder for facades (in drawing_and_style_transfer folder):
python train.py --dataroot=./datasets/facades/trainB --name=facades_autoencoder --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=2

In the reverse direction (images of facades):
python train.py --dataroot=./datasets/facades/trainA --name=facades_autoencoder_reverse --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=2

### Train OST

To train OST for images to facades:
python train.py --dataroot=./datasets/facades/ --name=facades_ost --load_dir=facades_autoencoder --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1

To train OST for facades to images (reverse direction):
python train.py --dataroot=./datasets/facades/ --name=facades_ost_reverse --load_dir=facades_autoencoder_reverse --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1 --A='B' --B='A'

To visualize losses: run python -m visdom.server

### Test OST

To test OST for images to facades:
python test.py --dataroot=./datasets/facades/ --name=facades_ost --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1

To test OST for facades to images (reverse direction):
python test.py --dataroot=./datasets/facades/ --name=facades_ost_reverse --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1 --A='B' --B='A'

### Options
Additional scripts for other datasets are at ./drawing_and_style_transfer/scripts

Options are at ./drawing_and_style_transfer/options

## Reference
If you found this code useful, please cite the following paper:
```
@inproceedings{Benaim2018OneShotUC,
  title={One-Shot Unsupervised Cross Domain Translation},
  author={Sagie Benaim and Lior Wolf},
  booktitle={NeurIPS},
  year={2018}
}
```

