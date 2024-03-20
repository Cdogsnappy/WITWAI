import os
import torch
import torch.nn as nn
import itertools

f = open(str(os.getcwd()) + "\Data\country_list.csv", 'r')
classes = [l.rstrip() for l in f]
classes = classes[1:]


class FinalFFNN(nn.Module):
    def __init__(self):
        super(FinalFFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(classes))
        )
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    # Forward takes a list of data, with x[1] the raw image data and x[0] the additional OCR and seg_net data.
    def forward(self, x):
        x = x[0] + self.conv_net(x[1])
        softmax = nn.Softmax(dim=1)
        return softmax(self.model(x))


epochs = 100
fin_model = FinalFFNN()
img_frame = ""
optimizer = torch.optim.Adam(itertools.chain(fin_model.model.parameters(), fin_model.conv_net.parameters()), lr=0.001)
criterion = nn.MSELoss()
for e in range(epochs):
    for img_batch in img_frame:
        print("This is where the training loop goes!")


