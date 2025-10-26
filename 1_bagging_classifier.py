# import libraries
from sklearn.ensemble import BaggingClassifier #torbolama modeli
from sklearn.tree import DecisionTreeClassifier #torbalama icinde kullanilacak  olan weak learner
from sklearn.datasets import load_iris # kullanacagimiz veri seti
from sklearn.model_selection import train_test_split #train test split fonksiyonu
from sklearn.metrics import accuracy_score# metrik dogruluk


# load dateset ->iris veri seti
iris = load_iris()
X = iris.data # features
y = iris.target # target variable
# data train and text split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# define base model: decision tree
base_model = DecisionTreeClassifier(random_state = 42)

# create bagging model
bagging_model = BaggingClassifier(
     estimator=base_model, # temel model:karar agaci
     n_estimators=10, # kullanilacak model sayisi
     max_samples=0.8, # her modeiln kullanacagi ornek orani
     max_features=0.8, # her modelin kullanacagi ozellik orani
     bootstrap=True, # orneklerin tekrar secilip secilmesine izin ver
     random_state= 42 # tekrarlanabilirlik icin bir sit, yani sabit bir tohum belirlendi
     )

# model training
bagging_model.fit(X_train,y_train)

# model testing
y_pred = bagging_model.predict(X_test)

# evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
