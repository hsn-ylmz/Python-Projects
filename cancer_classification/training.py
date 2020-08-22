

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from network import NetWork
import config
from imutils import paths
import numpy as np
import os


epochs = 40
init_lr = 1e-2
batch = 32

trainpath = list(paths.list_images(config.train_path))
len_train = len(trainpath)
len_validation = len(list(paths.list_images(config.validation_path)))
len_test = len(list(paths.list_images(config.test_path)))

train_label = [int(r.split(os.path.sep)[-2]) for r in trainpath]
train_label = np_utils.to_categorical(train_label)
classes = train_label.sum(axis=0)
class_weights = classes.max()/classes

train_gen_opt = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

validation_gen_opt = ImageDataGenerator(rescale=1/255)

train_gen = train_gen_opt.flow_from_directory(
    config.train_path,
    class_mode='categorical',
    target_size=(48, 48),
    color_mode='rgb',
    shuffle=True,
    batch_size=batch
)


validation_gen = validation_gen_opt.flow_from_directory(
    config.validation_path,
    class_mode='categorical',
    target_size=(48, 48),
    color_mode='rgb',
    shuffle=False,
    batch_size=batch
)

test_gen = validation_gen_opt.flow_from_directory(
    config.test_path,
    class_mode='categorical',
    target_size=(48, 48),
    color_mode='rgb',
    shuffle=False,
    batch_size=batch
)

model = NetWork.build(width=48, height=48, depth=3, classes=2)
optimizer = Adagrad(lr=init_lr, decay=init_lr/epochs)
model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

m = model.fit_generator(
    train_gen,
    steps_per_epoch=len_train//batch,
    validation_data=validation_gen,
    validation_steps=len_validation//batch,
    class_weight=class_weights,
    epochs=epochs
)

print("Evaluating the Model")
test_gen.reset()
pred_indices = model.predict_generator(test_gen, steps=(len_test//batch)+1)

pred_indices = np.argmax(pred_indices, axis=1)

print(classification_report(test_gen.classes,
                            pred_indices,
                            target_names=test_gen.class_indices.keys()
                            ))

cm = confusion_matrix(test_gen.classes, pred_indices)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
specificity = cm[1, 1]/(cm[1, 0] + cm[1, 1])
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print(cm)
print(f"Accuracy: {accuracy}")
print(f"Specificity: {specificity}")
print(f"Sensitivity: {sensitivity}")
