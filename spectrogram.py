import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import model_ResNet18
import model_ResNet50
import model_cider
from torch.utils.data import DataLoader
from spectrogramDatasets import SpectrogramDatasets, transform
from utils import set_logger, try_gpu
import warnings
warnings.filterwarnings('ignore')

model_name = sys.argv[1]
epochs = 200
batch_size = 8
current_filename = os.path.basename(__file__)[:-3] + "_" + model_name
logger = set_logger(filename=f"./log/log_{current_filename}.log")

if model_name == 'ResNet18':
    model = model_ResNet18.ResNet18(inputs_inchannel=3, fc_in_features=59904)
elif model_name == 'ResNet50':
    model = model_ResNet50.ResNet50(inputs_inchannel=3, fc_in_features=2048)
elif model_name == 'Cider':
    model = model_cider.Cider(inputs_inchannel=3, fc_in_features=1296)
model.to(try_gpu())
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

train_dataset = SpectrogramDatasets(
    "./data/spectrograms_split/train/",
    transform=transform
)
test_dataset = SpectrogramDatasets(
    "./data/spectrograms_split/val/",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size
)
test_loader = DataLoader(
    test_dataset,
    shuffle=True,
    batch_size=batch_size
)


def train(epoch):
    """训练过程"""
    total_loss = 0.0
    f1_scores = 0.0
    uar = 0.0
    auc = 0.0
    num = 0
    acc = 0.0
    for data in train_loader:
        inputs, target = data
        optimizer.zero_grad()
        inputs = inputs.to(try_gpu())
        target = target.to(try_gpu())
        outputs = model(inputs)
        outputs = outputs.to(torch.float32)
        target = target.to(torch.float32)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = []
        for o in outputs:
            pre = 1 if o >= 0.5 else 0
            predicted.append(pre)
        predicted = torch.from_numpy(np.array(predicted))
        predicted = predicted.to(try_gpu())
        f1_scores += f1_score(predicted.cpu(), target.cpu(), average='binary')
        tn, fp, fn, tp = confusion_matrix(target.cpu(), predicted.cpu(), labels=[0, 1]).ravel()
        acc += (tp + tn) / (tp + tn + fp + fn)
        try:
            sensitivity = tp / (tp + fn)
            specificity = tn / (fp + tn)
            uar += (specificity + sensitivity) / 2.0
            auc += roc_auc_score(target.cpu(), outputs.cpu().detach().numpy())
        except ValueError:
            pass
        num += 1
    logger.info('[epoch%d] loss: %.3f acc: %.3f f1_score: %.3f uar: %.3f, auc: %.3f'
                % (epoch, total_loss, acc/num, f1_scores/num, uar/num, auc/num))


def test(epoch):
    """测试过程"""
    f1_scores = 0.0
    uar = 0.0
    acc = 0.0
    auc = 0.0
    num = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(try_gpu())
            labels = labels.to(try_gpu())
            outputs = model(images)
            predicted = []
            for o in outputs:
                pre = 1 if o >= 0.5 else 0
                predicted.append(pre)
            p = pd.DataFrame(predicted)
            predicted = torch.from_numpy(np.array(predicted))
            predicted = predicted.to(try_gpu())
            predicted_temp = predicted.unsqueeze(0)
            labels_temp = labels.unsqueeze(0)
            f1_scores += f1_score(predicted.cpu(), labels.cpu(), average='binary')
            tn, fp, fn, tp = confusion_matrix(labels.cpu(), predicted.cpu(), labels=[0, 1]).ravel()
            acc += (tp + tn) / (tp + tn + fp + fn)
            try:
                sensitivity = tp / (tp + fn)
                specificity = tn / (fp + tn)
                uar += (specificity + sensitivity) / 2.0
                auc += roc_auc_score(labels.cpu(), outputs.cpu().detach().numpy())
            except ValueError:
                pass
            num += 1

    logger.info('(Test set) acc: %.6f  f1_score: %.6f uar: %.6f auc: %.6f'
                 %(acc/num, f1_scores/num, uar/num, auc/num))
    return acc/num, f1_scores/num,  uar/num, auc/num


if __name__ == "__main__":
    acc_max = 0
    acc_test = 0
    f1_max = 0
    uar_max = 0
    auc_max = 0
    for epoch in range(epochs):
        train(epoch)
        acc, f1, uar, auc = test(epoch)
        if acc > acc_max:
            torch.save(model.state_dict(), f'./model/{current_filename}_epoch' + str(epoch) + '.pt')
            acc_max = acc
        elif acc>=180 and acc > acc_test:
            torch.save(model.state_dict(), f'./model/{current_filename}_epoch' + str(epoch) + '.pt')
            acc_test = acc
        f1_max = f1 if f1 > f1_max else f1_max
        uar_max = uar if uar > uar_max else uar_max
        auc_max = auc if auc > auc_max else auc_max
    print(f'acc:{acc_max}, f1_score:{f1_max}, uar:{uar_max}, uac:{auc_max}')
