import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps

class path_to_img(Dataset):
    def __init__(self, img_path, labels, transform):  # 데이터셋 전처리
        self.img_path = img_path
        self.labels = labels
        self.transform = transform

    def __len__(self):  # 데이터셋 길이 (총 샘플의 수)
        return len(self.img_path)

    def __getitem__(self, idx):  # 데이터셋에서 특정 샘플을 가져옴
        image = Image.open('.' + self.img_path.iloc[idx])
        image = self.transform(image)
        return image, self.labels.iloc[idx]