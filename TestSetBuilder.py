import numpy as np
import os
import shutil
import pandas as pd
from PIL import Image

shutil.rmtree(str(os.getcwd()) + '/Data/test_data')
shutil.rmtree(str(os.getcwd()) + '/Data/train_data')
os.mkdir(str(os.getcwd()) + '/Data/test_data')
os.mkdir(str(os.getcwd()) + '/Data/train_data')
data = np.arange(9250)
test_set = np.sort(np.random.choice(data, 250, replace=False))
train_set = np.delete(data, test_set)
ocr_data = pd.read_csv(str(os.getcwd()) + '/Data/dataset/Pruned_Language_Predictions.csv')
test_table = ocr_data.iloc[test_set, 1:]
train_table = ocr_data.iloc[train_set, 1:]
test_table.to_csv(str(os.getcwd()) + '/Data/test_data/test_ocr.csv')
train_table.to_csv(str(os.getcwd()) + '/Data/train_data/train_ocr.csv')
avg_map = np.empty(3)
square_map = np.empty(3)
training_mean = np.array([119.24565898, 130.68539189, 133.48247856])
for val in test_set:
    shutil.copyfile(str(os.getcwd()) + '/seg_data/' + str(val) + '_segmented.png',
                    str(os.getcwd()) + '/Data/test_data/' + str(val) + '_segmented.png')
for val in train_set:
    shutil.copyfile(str(os.getcwd()) + '/seg_data/' + str(val) + '_segmented.png',
                    str(os.getcwd()) + '/Data/train_data/' + str(val) + '_segmented.png')
    img = np.array(Image.open(str(os.getcwd()) + '/resized_mapillary/' + str(val) + '.jpg').convert('RGB'))
    avg_map += np.sum(img, axis=(0, 1))
    square_map += np.sum(np.square(img-training_mean), axis=(0, 1))
pix_cnt = len(train_set) * (1024 ** 2)
mean = avg_map / pix_cnt
var = square_map / pix_cnt
std = np.sqrt(var)
img_stats = open('outputs/image_stats', 'w')
img_stats.write("Training Data Mean: " + str(mean) + "\nTraining Data Variance: " + str(
    var) + "\nTraining Data Standard Deviation: " + str(std))
img_stats.close()
