import numpy as np
import pandas as pd
import torch
import model_ResNet18
import model_ResNet50
import model_cider
from torch.utils.data import DataLoader
from spectrogramDatasets import SpectrogramDatasets, transform
from attack.fgsm import FGSM
from attack.pgd import PGD
import os
import sys
import imageio
from utils import set_logger, try_gpu
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


batch_size = 4
attack_model = sys.argv[1]
attack_method = sys.argv[2]
logger = set_logger(filename=f"./log/log_{attack_model}_{attack_method}.log")
device = try_gpu()

if attack_model == 'ResNet18':
    model = model_ResNet18.ResNet18(inputs_inchannel=3, fc_in_features=59904)
    model.to(device)
    model.load_state_dict(torch.load("./model/spectrogramResNet18_epoch180.pt", map_location="cpu"))
elif attack_model == 'ResNet50':
    model = model_ResNet50.ResNet50(inputs_inchannel=3, fc_in_features=2048)
    model.to(device)
    model.load_state_dict(torch.load("./model/spectrogram_ResNet50_epoch24.pt", map_location="cpu"))
elif attack_model == 'Cider':
    model = model_cider.Cider(inputs_inchannel=3, fc_in_features=1296)
    model.to(device)
    model.load_state_dict(torch.load("./model/spectrogramCider_epoch56.pt", map_location="cpu"))


test_dataset = SpectrogramDatasets(
    "./data/spectrograms_split/val/",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    shuffle=True,
    batch_size=batch_size
)


eps = [0, 8/255, 16/255, 32/255, 64/255]
accuracies = []
attack_accs = []
anao = []
k=0
for i in range(len(eps)):
    if i == 0:
        continue
    # print(eps[i])
    adv_examples = []
    if attack_method == 'fgsm':
        atk = FGSM(model, eps[i])
    elif attack_method == 'pgd':
        atk = PGD(model, eps[i])
    correct = 0
    total = 0
    l2norm = 0.0

    for images, labels in test_loader:
        adv_images = atk(images, labels)
        if i == 1:
            for adv_image in adv_images:
                adv_image = adv_image.swapaxes(0,1).swapaxes(1,2).cpu()
                lab = labels[k % batch_size]
                if not os.path.exists(f'./data/adv_spec/{attack_method}_{attack_model}/'):
                    os.makedirs(f'./data/adv_spec/{attack_method}_{attack_model}/')
                goal_path = f'./data/adv_spec/{attack_method}_{attack_model}/{k}_{lab}.png'
                anao.append([f'{k}_{lab}.png', lab.item()])
                imageio.imwrite(goal_path, adv_image)
                k += 1
        labels = labels.to(device)
        outputs = model(adv_images)
        final_pred = torch.tensor(np.array([int(x+0.5) for x in outputs.data])).to(device)
        total += 1
        correct += (final_pred == labels).sum()
        l2norm += np.linalg.norm(images.cpu() - adv_images.cpu())

    total *= batch_size
    acc = float(correct)/total
    print('epsilon: %.4f, Robust accuracy: %d/%d=%.6f, attack success rate:%.6f, dist(l2n):%.6f' 
          % (eps[i], correct, total, acc, 1-float(correct)/total, l2norm/total))
    accuracies.append(acc)
    attack_accs.append(1-acc)
    
anao = pd.DataFrame(anao)   
anao.to_csv(f'./data/adv_spec/{attack_method}_{attack_model}/annotations.csv', index=None, header=None)