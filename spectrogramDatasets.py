from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

batch_size = 16


class SpectrogramDatasets(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_labels = pd.read_csv(f"{root_dir}annotations.csv")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        # img_path = os.path.join(self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        # reverse_list = [1, 0]
        # label = reverse_list[label]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SpectrogramDatasets(
    "./data/spectrograms_split/train/",
    transform=transform
)
test_dataset = SpectrogramDatasets(
    "./data/spectrograms_split/val/",
    transform=transform
)
# train_dataset = SpectrogramDatasets(
#     "./attack_train_specs_split/train/",
#     transform=transform
# )
# test_dataset = SpectrogramDatasets(
#     "./attack_train_specs_split/val/",
#     transform=transform
# )
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