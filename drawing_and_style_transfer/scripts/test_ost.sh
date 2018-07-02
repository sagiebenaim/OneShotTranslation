# images to cityscapes
python test.py --dataroot=./datasets/cityscapes/ --name=cityscapes_ost --load_dir=cityscapes_ost --model=ost --no_dropout --n_downsampling=3 --num_unshared=3 --start=0 --max_items_A=1
# cityscapes to images
python test.py --dataroot=./datasets/cityscapes/ --name=cityscapes_ost_reverse --load_dir=cityscapes_ost_reverse --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1 --A='B' --B='A'

# images to facades
python test.py --dataroot=./datasets/facades/ --name=facades_ost --load_dir=facades_ost --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1
# facades to images
python test.py --dataroot=./datasets/facades/ --name=facades_ost_reverse --load_dir=facades_ost_reverse --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1 --A='B' --B='A'

# aerial view to maps
python test.py --dataroot=./datasets/maps/ --name=maps_ost --load_dir=maps_ost --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1
# maps to aerial view
python test.py --dataroot=./datasets/maps/ --name=maps_ost_reverse --load_dir=maps_ost_reverse --model=ost --no_dropout --n_downsampling=2 --num_unshared=2 --start=0 --max_items_A=1 --A='B' --B='A'

# monet2photo
python test.py --dataroot=./datasets/monet2photo/ --name=monet2photo_ost --load_dir=monet2photo_ost --model=ost --no_dropout --n_downsampling=2 --num_unshared=0 --start=0 --max_items_A=1
# photo2monet
python test.py --dataroot=./datasets/monet2photo/ --name=monet2photo_ost_reverse --load_dir=monet2photo_ost_reverse --model=ost --no_dropout --n_downsampling=2 --num_unshared=0 --start=0 --max_items_A=1 --A='B' --B='A'

# summer2winter
python test.py --dataroot=./datasets/summer2winter_yosemite/ --name=summer2winter_yosemite_ost --load_dir=summer2winter_yosemite_ost --model=ost --no_dropout --n_downsampling=2 --num_unshared=0 --start=0 --max_items_A=1
# winter2summer
python test.py --dataroot=./datasets/summer2winter_yosemite/ --name=summer2winter_yosemite_ost_reverse --load_dir=summer2winter_yosemite_ost_reverse --model=ost --no_dropout --n_downsampling=2 --num_unshared=0 --start=0 --max_items_A=1 --A='B' --B='A'