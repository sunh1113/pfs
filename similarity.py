from attacks.blended import Blended
from datasets.cifar10 import CIFAR10
import torchvision.transforms as transforms
import numpy as np
from models import build_model

import torch


transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

path = 'r18_c10_model.pth'
net = build_model('r18', 10)
net = net.cuda()
checkpoint = torch.load(path)
net.load_state_dict(checkpoint['net'])
net.eval()

trigger = Blended(32, 50000)
backdoor_data = CIFAR10('../cifar_class/data/', True, trigger, transform_train)
original_data = CIFAR10('../cifar_class/data/', True, None, transform_train)

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
         raise RuntimeError("array dimensions {} not right".format(a.ndim))
    aa = np.squeeze(np.sum(a*b, axis=1))
    bb = np.squeeze(a_norm * b_norm)
    similiarity = aa/bb
    return similiarity

def distance(sample_idx):
    backdoor_datas = []
    original_datas = []
    for id in sample_idx:
        data, _, _, _, _ = backdoor_data.__getitem__(id)
        backdoor_datas.append(data)
        data, _, _, _, _ = original_data.__getitem__(id)
        original_datas.append(data)

    backdoor_datas = torch.stack(backdoor_datas, 0)
    original_datas = torch.stack(original_datas, 0)

    backdoor_datas = backdoor_datas.cuda()
    original_datas = original_datas.cuda()

    backdoor_feature = net(backdoor_datas, return_step=True).cpu().detach().numpy()
    original_feature = net(original_datas, return_step=True).cpu().detach().numpy()

    similiarity = 1.0 - cosine_distance(backdoor_feature, original_feature)
    return similiarity


distance_1 = []
for i in range(500):
    print(i)
    sample_idx = []
    for j in range(i * 100, (i + 1) * 100):
        idx = j
        sample_idx.append(idx)
    distance_2 = distance(sample_idx)
    distance_1 = np.concatenate((distance_1, distance_2), axis=0)


print(len(distance_1), distance_1.shape)
np.save('./distance/r18_c10_blend.npy', distance_1)