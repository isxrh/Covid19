import os
import glob
import shutil
import splitfolders
import pandas as pd
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


def prepare_dataset():
    """合并原始样本与对抗样本形成"""

    # 对抗样本
    for dir in os.listdir('./data/adv_spec'):
        for file in os.listdir(f'./data/adv_spec/{dir}/'):
            if file[-1] != 'v' and file[-5] == '0':
                shutil.copyfile(f'./data/adv_spec/{dir}/{file}', f'./data/attack_train_specs/negative/{dir}_{file}')
            elif file[-1] != 'v' and file[-5] == '1':
                shutil.copyfile(f'./data/adv_spec/{dir}/{file}', f'./data/attack_train_specs/positive/{dir}_{file}')

    # 原始样本
    for dir1 in os.listdir('./data/spectrograms_split/'):
        for dir2 in os.listdir(f'./data/spectrograms_split/{dir1}/'):
            if dir2 == 'negative':
                for file in os.listdir(f'./data/spectrograms_split/{dir1}/{dir2}/'):
                    shutil.copyfile(f'./data/spectrograms_split/{dir1}/{dir2}/{file}', f'./data/attack_train_specs/negative/{file[:-4]}_0.png')
            elif dir2 == 'positive':
                for file in os.listdir(f'./data/spectrograms_split/{dir1}/{dir2}/'):
                    shutil.copyfile(f'./data/spectrograms_split/{dir1}/{dir2}/{file}', f'./data/attack_train_specs/positive/{file[:-4]}_1.png')

    # 统计正负样本数目
    positive_spec = [spec for spec in os.listdir("./data/attack_train_specs/positive/")]
    negative_spec = [spec for spec in os.listdir("./data/attack_train_specs/negative/")]
    print(f"Number of Negative spectrograms: {len(positive_spec)}  \nNumber of Positive spectrograms: {len(negative_spec)}")

    # 划分训练集和测试集
    RATIO = (0.8, 0.2)
    splitfolders.ratio("./data/attack_train_specs", output="./data/attack_train_specs_split",
                       seed=1337, ratio=RATIO)


    def create_annotations_csv(file_path):
        """创建注释文件"""
        spec_list = []
        labels = ['negative', 'positive']
        for label in labels:
            for spec in glob.glob(f"{file_path}{label}/*"):
                spec_list.append([spec, labels.index(label), label])
        spec_list = pd.DataFrame(spec_list)
        spec_list.to_csv(f'{file_path}annotations.csv', index=None, header=None)
    create_annotations_csv("./data/attack_train_specs_split/train/")
    create_annotations_csv("./data/attack_train_specs_split/val/")


model_name = sys.argv[1]
epochs = 200
batch_size = 8
current_filename = os.path.basename(__file__)[:-3] + "_" + model_name
logger = set_logger(filename=f"./log/log_{current_filename}_attack.log")

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
    "./data/attack_train_specs_split/train/",
    transform=transform
)
test_dataset = SpectrogramDatasets(
    "./data/attack_train_specs_split/val/",
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
    prepare_dataset()
    acc_max = 0
    acc_test = 0
    f1_max = 0
    uar_max = 0
    auc_max = 0
    for epoch in range(epochs):
        train(epoch)
        acc, f1, uar, auc = test(epoch)
        f1_max = f1 if f1 > f1_max else f1_max
        uar_max = uar if uar > uar_max else uar_max
        auc_max = auc if auc > auc_max else auc_max
    print(f'acc:{acc_max}, f1_score:{f1_max}, uar:{uar_max}, uac:{auc_max}')
