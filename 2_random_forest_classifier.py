# import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# load dataset : breast cancer veri seti
cancer = load_breast_cancer()
X = cancer.data #features: orn, tumor boyutu, sekli, alanı...
y = cancer.target #hedef degisken 0:malignant(kotu huylu), 1: benign(iyi huylu)    
# veriyi train ve test veri setleri oalrak ikiye ayır;train(egitim verisi) test(egitilmis modelin testi verisi)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
 
# create random forest model
rf_model = RandomForestClassifier(
    n_estimators =100,#agac sayisi
    max_depth =10,#maksimum derinlik,agaclarin inebilecegi derinlik(dallardan veya koklerden dolayidir.cok derin yaparsak ogrenmeyi engellemis oluruz)
    min_samples_split=5, # bir dugumu bolmek icin minimum ornek sayisi
    random_state=42
    )

# training
rf_model.fit(X_train, y_train)

# testing
y_pred = rf_model.predict(X_test)

# evaluation: accuracy ve classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
