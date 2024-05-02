import os

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from final_net import FinalFFNN
from CustomImageDataset import CustomImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(os.path.dirname(os.path.dirname(os.getcwd())) + str('/outputs/saved_models/model1.pt'))
model = nn.DataParallel(model, device_ids=[0])


f = open(str(os.path.dirname(os.path.dirname(os.getcwd()))) + "/Data/used_countries.csv", 'r')
classes = [l.rstrip() for l in f]
classes = classes[1:]
smoothing_factor = .01
test_data = CustomImageDataset('../Data/test_data/test_ocr.csv', 'Data/test_data')
test_loader = DataLoader(test_data, batch_size=len(classes) * 10, shuffle=False)
true_label_matrix = np.diag(np.ones(len(classes)) * (1 - len(classes) * smoothing_factor))
true_label_matrix += smoothing_factor

test_labels, test_data = next(iter(test_loader))
true_label = true_label_matrix[test_labels.numpy(), :]
test_images, test_lang = test_data
model.eval()
pred = Tensor.cpu(model.forward([test_lang.to(device), test_images.to(device)])).detach().numpy()  # (10xlen(classes))xlen(classes) tensor representing probability of each country for the test data
country_prediction = np.argmax(pred, 1)
real_prediction = np.argmax(true_label, 1)
for val in range(len(country_prediction)):
    print('Guess: ' + str(classes[country_prediction[val]]) + ' Truth: ' + str(classes[real_prediction[val]]))
accuracy = (len(classes)*10 - np.count_nonzero(country_prediction - real_prediction)) / (len(classes)*10)
print("Accuracy of " + str(accuracy) + " achieved!")
