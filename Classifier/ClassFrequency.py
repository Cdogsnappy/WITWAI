import os

import pandas as pd
import numpy as np

f = open(str(os.path.dirname(os.getcwd())) + "\Data\country_list.csv", 'r')
classes = [l.rstrip() for l in f]
classes = classes[1:]

ocr_data = pd.read_csv(str(os.path.dirname(os.getcwd())) + '/Data/dataset/Pruned_Language_Predictions.csv')
label_list = ocr_data.iloc[:, 3]
label_freq = np.ones(len(classes))
for val in label_list:
    label_freq[classes.index(val)] += 1
class_freq = open('../outputs/class_freq', 'w')
class_freq.write(str(9250/(len(classes)*label_freq)))
class_freq.close()
