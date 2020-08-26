
import pandas
import numpy
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer

train_data = pandas.read_csv("train.csv")
test_data = pandas.read_csv("test.csv")
gender_sub = pandas.read_csv("gender_submission.csv")


x_train = train_data[["Pclass", "Sex", "Age", "SibSp",
                "Parch", "Fare", "Embarked"]]
y_train = train_data[["Survived"]]

x_test = test_data[["Pclass", "Sex", "Age", "SibSp",
                "Parch", "Fare", "Embarked"]]
y_test = gender_sub[["Survived"]]

label_encode = LabelEncoder()
label_binarize = LabelBinarizer()
imputer_mean = SimpleImputer(missing_values=numpy.nan, strategy="mean")
imputer_median = SimpleImputer(missing_values=numpy.nan, strategy="median")

#Impute Age and Fare
x_train["Age"] = imputer_mean.fit_transform(x_train[["Age"]]).round(0)
x_test["Age"] = imputer_mean.fit_transform(x_test[["Age"]]).round(0)
x_test["Fare"] = imputer_mean.fit_transform(x_test[["Fare"]])

#Turn string to integer and Impute Embark
embarks = {"Embarked": {"S": 0, "C": 1, "Q": 2}}
x_train.replace(embarks, inplace = True)
x_train["Embarked"] = imputer_median.fit_transform(x_train[["Embarked"]])
x_test.replace(embarks, inplace=True)
x_test["Embarked"] = imputer_median.fit_transform(x_test[["Embarked"]])

#Encode Sex column as 0 and 1
x_train["Sex"] = label_encode.fit_transform(x_train["Sex"])
x_test["Sex"] = label_encode.fit_transform(x_test["Sex"])

#Turn integer data to categorical data
x_train["Embarked"] = pandas.Categorical(x_train["Embarked"])
x_test["Embarked"] = pandas.Categorical(x_test["Embarked"])

#Encode as binary and drop Embarked column
binarized_labels = label_binarize.fit_transform(x_train["Embarked"])
x_train[["S", "C", "Q"]] = binarized_labels
x_train = x_train.drop(columns = ["Embarked"])

binarized_labels_test = label_binarize.fit_transform(x_test["Embarked"])
x_test[["S", "C", "Q"]] = binarized_labels_test
x_test = x_test.drop(columns = ["Embarked"])

