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

if not os.path.exists(os.path.dirname(os.path.dirname(os.getcwd())) + str('/outputs')):
    os.mkdir(os.path.dirname(os.path.dirname(os.getcwd())) + '/outputs')
print(torch.seed())
np.set_printoptions(threshold=np.inf)
f = open(str(os.path.dirname(os.path.dirname(os.getcwd()))) + "/Data/used_countries.csv", 'r')
classes = [l.rstrip() for l in f]
classes = classes[1:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FinalFFNN(nn.Module):
    def __init__(self):
        super(FinalFFNN, self).__init__()
        self.ocr_net = nn.Sequential(
            nn.Linear(35, 48),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(20232, 10116),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(10116, 5058),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(5058, 2048),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, len(classes))
        )
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 24, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    # Forward takes a list of data, with x[1] the image data and x[0] the additional OCR data.
    def forward(self, x):
        conv_out = self.conv_net(x[1])
        nn_in = torch.cat((torch.flatten(conv_out, start_dim=1), self.ocr_net(x[0].float())), 1)
        softmax = nn.Softmax(dim=1)
        return softmax(self.model(nn_in.float()))


def run():
    epochs = 4
    batch_size = 10
    smoothing_factor = .01
    fin_model = FinalFFNN()
    fin_model = nn.DataParallel(fin_model, device_ids=[0])
    fin_model = fin_model.to(device)
    optimizer = torch.optim.Adam(
        itertools.chain(fin_model.module.model.parameters(), fin_model.module.conv_net.parameters()), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    losses = []
    train_data = CustomImageDataset('../Data/train_data/train_ocr.csv', 'Data/train_data')
    test_data = CustomImageDataset('../Data/test_data/test_ocr.csv', 'Data/test_data')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(classes) * 10, shuffle=False)
    true_label_matrix = np.diag(np.ones(len(classes)) * (1 - len(classes) * smoothing_factor))
    true_label_matrix += smoothing_factor
    for e in range(epochs):
        loss_e = 0.0
        for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
            real, _ = data
            images, lang_vec = _
            true_label = torch.from_numpy(true_label_matrix[real.numpy(), :]).to(device)
            optimizer.zero_grad()
            pred = fin_model.forward([lang_vec.to(device), images.to(device)]).to(device)
            loss = criterion(pred, true_label)
            loss.backward()
            optimizer.step()
            loss_e += loss.item()
        print(loss_e / bi)
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
    fin_model.eval()
    pred = Tensor.cpu(fin_model.forward([test_lang.to(device), test_images.to(
        device)])).detach().numpy()  # (10xlen(classes))xlen(classes) tensor representing probability of each country for the test data
    country_prediction = np.argmax(pred, 1)
    real_prediction = np.argmax(true_label, 1)
    for val in range(len(country_prediction)):
        print('Guess: ' + str(classes[country_prediction[val]]) + ' Truth: ' + str(classes[real_prediction[val]]))
    accuracy = (len(classes) * 10 - np.count_nonzero(country_prediction - real_prediction)) / (len(classes) * 10)
    print("Accuracy of " + str(accuracy) + " achieved!")
    torch.save(fin_model, os.path.dirname(os.path.dirname(os.getcwd())) + str('/outputs/model.pt'))
