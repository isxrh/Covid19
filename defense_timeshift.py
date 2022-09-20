import numpy as np
import torch
import model_ResNet18
import model_ResNet50
import model_cider
from utils import try_gpu
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import sys
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

batch_size = 4
attack_model = sys.argv[1]
attack_method = sys.argv[2]
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


class SpectrogramDatasets(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_labels = pd.read_csv(f"{root_dir}annotations.csv")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        frame_num = image.shape[-1]
        max_shifts = frame_num * 3
        nb_shifts = np.random.randint(-max_shifts, max_shifts)
        image = np.roll(image, nb_shifts, axis=2)
        label = self.img_labels.iloc[idx, 1]
        return image, label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SpectrogramDatasets(
    f"./data/adv_spec/{attack_method}_{attack_model}/",
    transform=transform
)

loader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=batch_size
)


correct = 0
total = 0
for images, labels in loader:
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    outputs = model(images)
    pre = torch.tensor(np.array([int(x+0.5) for x in outputs.data]))
    correct += (pre.cpu() == labels.cpu()).sum()
    total += 1
total *= batch_size
print(f'model acc: {correct}/{total}={correct/float(total)}')
