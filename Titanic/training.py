from keras.optimizers import Adagrad
from sklearn.metrics import classification_report, confusion_matrix
from NetWork import NetWork
import preprocessing as pp
import pandas

x_train = pp.x_train
x_test = pp.x_test
y_train = pp.y_train
y_test = pp.y_test

epochs = 100
lr = 0.1
batch = 16


model = NetWork.build()
optimizer = Adagrad(lr=lr, decay = lr/epochs)
model.compile(loss= "binary_crossentropy",
              optimizer = optimizer,
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch, epochs=epochs)


y_pred = pandas.DataFrame(model.predict(x_test))
y_pred.values[y_pred.values > 0.5] = 1
y_pred.values[y_pred.values <= 0.5] = 0

cm = confusion_matrix(y_test, y_pred)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
spec = cm[1, 1] / (cm[1, 0] + cm[1, 1])
sens = cm[0, 0] / (cm[0, 0] + cm[0, 1])


print(cm)
print(f"Accuracy: {acc}")
print(f"Specificity: {spec}")
print(f"Sensitivity: {sens}")