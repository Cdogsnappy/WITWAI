import numpy as np
import os
import shutil
import pandas as pd
from PIL import Image

data_size = 70
if os.path.exists(str(os.path.dirname(os.getcwd())) + '/Data/test_data'):
    shutil.rmtree(str(os.path.dirname(os.getcwd())) + '/Data/test_data')
if os.path.exists(str(os.path.dirname(os.getcwd())) + '/Data/train_data'):
    shutil.rmtree(str(os.path.dirname(os.getcwd())) + '/Data/train_data')
if os.path.exists(str(os.path.dirname(os.getcwd())) + '/Data/used_countries.csv'):
    os.remove(str(os.path.dirname(os.getcwd())) + '/Data/used_countries.csv')
os.mkdir(str(os.path.dirname(os.getcwd())) + '/Data/test_data')
os.mkdir(str(os.path.dirname(os.getcwd())) + '/Data/train_data')
ocr_data = pd.read_csv(str(os.path.dirname(os.getcwd())) + '/Data/dataset/Pruned_Language_Predictions.csv')
f = open(str(os.path.dirname(os.getcwd())) + "/Data/country_list.csv", 'r')
g = open(str(os.path.dirname(os.getcwd())) + '/Data/used_countries.csv', 'w')
g.write('country\n')
classes = [l.rstrip() for l in f]
classes = classes[1:]
train_set = np.empty(0)
test_set = np.empty(0)
for c in classes:
    potential_train_data = ocr_data.index[ocr_data['country'] == c]
    if len(potential_train_data) < data_size:
        print('skipped ' + str(c))
        continue
    data = np.arange(len(potential_train_data))
    train_c = np.random.choice(data, data_size, replace=False)
    train_data = np.array(potential_train_data[train_c[0:len(train_c) - 10]])
    test_data = np.array(potential_train_data[train_c[len(train_c) - 10:len(train_c)]])
    train_set = np.concatenate((train_set, train_data))
    test_set = np.concatenate((test_set, test_data))
    g.write(c + '\n')

np.sort(train_set)
np.sort(test_set)
test_table = ocr_data.iloc[test_set, 1:]
train_table = ocr_data.iloc[train_set, 1:]
test_table.to_csv(str(os.path.dirname(os.getcwd())) + '/Data/test_data/test_ocr.csv')
train_table.to_csv(str(os.path.dirname(os.getcwd())) + '/Data/train_data/train_ocr.csv')
for val in test_set:
    shutil.copyfile(str(os.path.dirname(os.getcwd())) + '/seg_data/' + str(int(val)) + '_segmented.png',
                    str(os.path.dirname(os.getcwd())) + '/Data/test_data/' + str(int(val)) + '_segmented.png')
for val in train_set:
    shutil.copyfile(str(os.path.dirname(os.getcwd())) + '/seg_data/' + str(int(val)) + '_segmented.png',
                    str(os.path.dirname(os.getcwd())) + '/Data/train_data/' + str(int(val)) + '_segmented.png')
    img = np.array(Image.open(str(os.path.dirname(os.getcwd())) + '/resized_mapillary/' + str(int(val)) + '.jpg').convert('RGB'))
