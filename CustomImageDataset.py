import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as TF

f = open(str(os.getcwd()) + "/Data/used_countries.csv", 'r')
classes = [l.rstrip() for l in f]
classes = classes[1:]
f = open(str(os.getcwd()) + "/Data/language_list.csv", 'r')
languages = [l.rstrip() for l in f]
transform = TF.Compose([TF.PILToTensor()])
mean = np.empty((3, 1024, 1024))
training_mean = np.array([119.29616104, 130.72161865, 133.50318999])
training_std = np.array([67.23823554, 69.06779112, 77.32858312])


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label, ocr_pred, conf = self.img_labels.iloc[idx, 3:6]
        image = np.array(Image.open(self.img_dir + "/" + str(self.img_labels.iloc[idx, 0]) + "_segmented.png").convert('RGBA'), dtype="float32")
        image[:, :, 0] = (image[:, :, 0] - training_mean[0]) / training_std[0]
        image[:, :, 1] = (image[:, :, 1] - training_mean[1]) / training_std[1]
        image[:, :, 2] = (image[:, :, 2] - training_mean[2]) / training_std[2]
        image = torch.tensor(np.reshape(image, (4,1024,1024)))
        lang_vec = np.zeros(len(languages))
        if isinstance(ocr_pred, str):
            lang_vec[languages.index(ocr_pred)] += (conf / 10.0)
        data = (image, torch.from_numpy(lang_vec))
        return classes.index(label), data
