import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('Data.txt')

#print(len(dataset))
#print(dataset.head())


#split dataset
X = dataset.iloc[:,0:9]
y = dataset.iloc[:,9]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size= 0.2)
