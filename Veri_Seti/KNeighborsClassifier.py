## K Neighbors  Classifier
from sklearn.datasets import load_iris
iris= load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.target)
print(iris.data)

X = iris.data
Y = iris.target

from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
print("eğitim veri seti boyutu : ",len(X_train))
print("test veri seti boyutu: ",len(X_test))
print("eğitim veri seti boyutu : ",len(Y_train))
print("test veri seti boyutu: ",len(Y_test))

from sklearn.neighbors  import  KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train,Y_train)

Y_tahmin=model.predict(X_test)
from sklearn.metrics import confusion_matrix
hata_matrisi=confusion_matrix(Y_test,Y_tahmin)
print(hata_matrisi)

import seaborn as sn
import pandas as pd
import  matplotlib.pyplot as  plt

index = ['setosa','versicolar','virginica']
Columns = ['setosa','versicolar','virginica']
hata_goster= pd.DataFrame(hata_matrisi,Columns,index)
plt.figure(figsize=(5,20))
sn.heatmap(hata_goster,annot=True)
plt.show()
