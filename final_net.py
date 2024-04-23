import os
import numpy as np
import torch
import torch.nn as nn
import itertools

from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

from torch.utils.data import DataLoader

from CustomImageDataset import CustomImageDataset

np.set_printoptions(threshold=np.inf)
f = open(str(os.getcwd()) + "\Data\country_list.csv", 'r')
classes = [l.rstrip() for l in f]
classes = classes[1:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FinalFFNN(nn.Module):
    def __init__(self):
        super(FinalFFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2387, 1024),
            nn.LeakyReLU(.1),
            nn.Linear(1024, 128),
            nn.LeakyReLU(.1),
            nn.Linear(128, len(classes))
        )
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(.1),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(.1),
            nn.Conv2d(32, 12, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    # Forward takes a list of data, with x[1] the image data and x[0] the additional OCR data.
    def forward(self, x):
        conv_out = self.conv_net(x[1])
        nn_in = torch.cat((torch.flatten(conv_out, start_dim=1), x[0]), 1)
        softmax = nn.Softmax(dim=1)
        return softmax(self.model(nn_in.float()))


epochs = 2
batch_size = 40
smoothing_factor = .02
fin_model = FinalFFNN()
fin_model = nn.DataParallel(fin_model, device_ids=[0])
fin_model = fin_model.to(device)
optimizer = torch.optim.Adam(
    itertools.chain(fin_model.module.model.parameters(), fin_model.module.conv_net.parameters()), lr=0.04)
criterion = nn.FocalLoss
criterion.to(device)
losses = []
train_data = CustomImageDataset('Data/train_data/train_ocr.csv', 'Data/train_data')
test_data = CustomImageDataset('Data/test_data/test_ocr.csv', 'Data/test_data')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=250, shuffle=False)
true_label_matrix = np.diag(np.ones(len(classes))*(1-len(classes)*smoothing_factor))
true_label_matrix+=smoothing_factor
print(true_label_matrix)
for e in range(epochs):
    loss_e = 0.0
    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
        real, _ = data
        images, lang_vec = _
        true_label = torch.from_numpy(true_label_matrix[real.numpy(), :]).to(device)
        optimizer.zero_grad()
        pred = fin_model.forward([lang_vec.to(device), images.to(device)]).to(device)
        #print(Tensor.cpu(pred).detach().numpy())
        loss = criterion(pred, true_label)
        loss.backward()
        optimizer.step()
        print(loss.item())
        loss_e += loss.item()
    print(loss_e/bi)
    losses.append(loss_e / bi)
plt.figure()
plt.plot(losses)
plt.title("Loss (Cross-Entropy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('outputs/loss.png')
test_labels, test_data = next(iter(test_loader))
true_label = true_label_matrix[test_labels.numpy(), :]
test_images, test_lang = test_data
pred = Tensor.cpu(fin_model.forward([test_lang.to(device), test_images.to(
    device)])).detach().numpy()  # 250xlen(classes) tensor representing probability of each country for the test data
country_prediction = np.argmax(pred, 1)
real_prediction = np.argmax(true_label, 1)
accuracy = (250 - np.count_nonzero(country_prediction - real_prediction)) / 250
print("Accuracy of " + str(accuracy) + " achieved!")
