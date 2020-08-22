
# Data can be found at
# "https://www.kaggle.com/paultimothymooney/breast-histopathology-images"

import os

home = os.path.expanduser("~")


original_input = os.path.join(home, "Breast_Histopathology/original")

data_set = os.path.join(home, "Breast_Histopathology/dataset")

train_path = os.path.sep.join([data_set, "training"])
validation_path = os.path.sep.join([data_set, "validation"])
test_path = os.path.sep.join([data_set, "testing"])


train_split = 0.8
validation_split = 0.1
