import argparse
import logging
import os

from data_loader import get_loader
from torch.backends import cudnn

from mnist_to_svhn.solver_svhn_to_mnist import Solver


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    svhn_loader, mnist_loader, svhn_test_loader, mnist_test_loader = get_loader(config)

    solver = Solver(config, svhn_loader, mnist_loader)
    cudnn.benchmark = True 
    
    # create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    base = config.log_path
    filename = os.path.join(base, str(config.max_items))
    if not os.path.isdir(base):
        os.mkdir(base)
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    
    if config.mode == 'train':
        solver.train()

    elif config.mode == 'sample':
        solver.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    
    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--kl_lambda', type=float, default=0.1)
    
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mnist_path', type=str, default='./mnist')
    parser.add_argument('--svhn_path', type=str, default='./svhn')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--shuffle', type=bool, default=True)

    parser.add_argument('--load_iter', type=int, default=10000)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--num_averaging_runs', type=int, default=1000)
    parser.add_argument('--num_iters_save_model_and_return', type=int, default=5000)
    parser.add_argument('--num_d_iterations', type=int, default=1)
    parser.add_argument('--num_g_iterations', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./models_ost')
    parser.add_argument('--sample_path', type=str, default='./samples_ost')
    parser.add_argument('--load_path', type=str, default='./models_autoencoder')
    parser.add_argument('--log_path', type=str, default='logs_ost')
    parser.add_argument('--pretrained_g', required=True, type=str2bool)
    parser.add_argument('--save_models_and_samples', required=True, type=str2bool)
    parser.add_argument('--use_augmentation', required=True, type=str2bool)
    parser.add_argument('--one_way_cycle', required=True, type=str2bool)
    parser.add_argument('--freeze_shared', required=True, type=str2bool)
    parser.add_argument('--max_items', type=int, default=1)

    config = parser.parse_args()
    print(config)
    main(config)