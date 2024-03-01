from sklearn.linear_model import LogisticRegression
import pandas as pd  
from sklearn.metrics import accuracy_score  
from sklearnex import patch_sklearn
patch_sklearn()

X = pd.read_csv("dataset/train.csv")
Y = X.drop('num_sold')

logreg = LogisticRegression(random_state = 0)  
logreg.fit(X, Y)  
Y_pred = logreg.predict(X)  
score = accuracy_score(Y, Y_pred)  
print(score) 