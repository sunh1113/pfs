from os.path import join
from datasets.cifar10 import CIFAR10
from datasets.cifar100 import CIFAR100
from datasets.tinyimagenet200 import TinyImageNet200
import torchvision.transforms as transforms

DATASETS = {
    'c10': CIFAR10,
    'c100': CIFAR100,
    'i200': TinyImageNet200,
}

def transform_set(transform_id, train, img_size, crop_pad, flip):
    transform = []
    transform.append(transforms.Resize((img_size, img_size)))

    if transform_id == 0:   # RandomCrop + RandomHorizontalFlip
        transform.append(transforms.Pad(crop_pad))
        if train:
            transform.append(transforms.RandomCrop((img_size, img_size)))
            if flip: transform.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            transform.append(transforms.CenterCrop((img_size, img_size)))

    # transform_id == 1, None

    if transform_id == 2:   # RandomCrop
        transform.append(transforms.Pad(crop_pad))
        if train:
            transform.append(transforms.RandomCrop((img_size, img_size)))
        else:
            transform.append(transforms.CenterCrop((img_size, img_size)))

    if transform_id == 3:   # RandomHorizontalFlip
        if train:
            transform.append(transforms.RandomHorizontalFlip(p=0.5))

    if transform_id == 4:   # RandomRotation
        if train:
            transform.append(transforms.RandomRotation(45))

    if transform_id == 5:   # ColorJitter
        if train:
            transform.append(transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.2, 0.2)))

    transform = transforms.Compose(transform)
    return transform

def build_data(data_name, data_path, train, trigger, transform):
    data = DATASETS[data_name](root=join(data_path, DATASETS[data_name].__name__.lower()), train=train, trigger=trigger, transform=transform)
    return data
