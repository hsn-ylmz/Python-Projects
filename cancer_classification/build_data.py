import config
from imutils import paths
import random
import shutil
import os


original_path = list(paths.list_images(config.original_input))
random.seed(5)
random.shuffle(original_path)

index = int(len(original_path)*config.train_split)
trainpath = original_path[:index]
testpath = original_path[index:]

index = int(len(trainpath)*config.validation_split)
validationpath = trainpath[:index]
trainpath = trainpath[index:]

dataset = [("training", trainpath, config.train_path),
           ("validation", validationpath, config.validation_path),
           ("test", testpath, config.test_path)
           ]

for (setType, original_path, base_path) in dataset:
    print(f"Building {setType} set")

    if not os.path.exists(base_path):
        print(f"Creating {base_path} directory")
        os.makedirs(base_path)

    for path in original_path:
        file = path.split(os.path.sep)[-1]
        label = file[-5:-4]

        label_path = os.path.sep.join([base_path, label])
        if not os.path.exists(label_path):
            print(f"Creating {label_path} directory")
            os.makedirs(label_path)

        new_path = os.path.sep.join([label_path, file])
        shutil.copy2(path, new_path)
