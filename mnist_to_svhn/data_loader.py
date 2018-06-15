import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    transform_list = []

    if config.use_augmentation:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomRotation(0.1))

    transform_list.append(transforms.Scale(config.image_size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform_test = transforms.Compose([
        transforms.Scale(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train = transforms.Compose(transform_list)

    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform_train, split='train')
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform_train, train=True)

    svhn_test = datasets.SVHN(root=config.svhn_path, download=True, transform=transform_test, split='test')
    mnist_test = datasets.MNIST(root=config.mnist_path, download=True, transform=transform_test, train=False)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=config.shuffle,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=config.shuffle,
                                               num_workers=config.num_workers)

    svhn_test_loader = torch.utils.data.DataLoader(dataset=svhn_test,
                                                   batch_size=config.batch_size,
                                                   shuffle=False,
                                                   num_workers=config.num_workers)

    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                                    batch_size=config.batch_size,
                                                    shuffle=False,
                                                    num_workers=config.num_workers)

    return svhn_loader, mnist_loader, svhn_test_loader, mnist_test_loader