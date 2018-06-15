# Implementation of One-Shot Unsupervised Cross Domain Translation

## MNIST-to-SVHN and SVHN-to-MNIST OST
Requirements:
1. Pytorch 0.4
2. Python 3.6

To train autoencoder for both MNIST and SVHN:
python ./mnist_to_svhn/main_autoencoder.py

To train OST for MNIST to SVHN:
python ./mnist_to_svhn/main_mnist_to_svhn.py --pretrained_g=True --save_models_and_samples=True --use_augmentation=True --one_way_cycle=True --freeze_shared=False

To train OST for SVHN to MNIST:
python ./mnist_to_svhn/main_svhn_to_mnist.py --pretrained_g=True --save_models_and_samples=True --use_augmentation=True --one_way_cycle=True --freeze_shared=False

## Drawing and Style Transfer Tasks

Will be uploaded shortly

