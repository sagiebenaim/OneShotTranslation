# images to cityscapes
python train.py --dataroot=./datasets/cityscapes/trainB --name=cityscapes_autoencoder --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=3 --num_unshared=3
# cityscapes to images
python train.py --dataroot=./datasets/cityscapes/trainA --name=cityscapes_autoencoder_reverse --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=2

# images to facades
python train.py --dataroot=./datasets/facades/trainB --name=facades_autoencoder --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=2
# facades to images
python train.py --dataroot=./datasets/facades/trainA --name=facades_autoencoder_reverse --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=2

# aerial view to maps
python train.py --dataroot=./datasets/maps/trainB --name=maps_autoencoder --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=2
# maps to aerial view
python train.py --dataroot=./datasets/maps/trainA --name=maps_autoencoder_reverse --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=2

# monet2photo
python train.py --dataroot=./datasets/monet2photo/trainB --name=monet2photo_autoencoder --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=0
# photo2monet
python train.py --dataroot=./datasets/monet2photo/trainA --name=monet2photo_autoencoder_reverse --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=0

# summer2winter
python train.py --dataroot=./datasets/summer2winter_yosemite/trainB --name=summer2winter_yosemite_autoencoder --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=0
# winter2summer
python train.py --dataroot=./datasets/summer2winter_yosemite/trainA --name=summer2winter_yosemite_autoencoder_reverse --model=autoencoder --dataset_mode=single --no_dropout --n_downsampling=2 --num_unshared=0