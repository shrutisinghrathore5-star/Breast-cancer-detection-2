
import pandas as pd #For Data Manipulation

data=pd.read_csv("data.csv") #load data
df=pd.DataFrame(data) #Turn into dataFrame
df=df.drop('id',axis=1) #Drop unnecessary columns
df=df.drop('Unnamed: 32',axis=1)



from sklearn.model_selection import train_test_split #To split training and testing data
from sklearn.preprocessing import StandardScaler #Standardisation: converting every feature to the same kind of scale
from sklearn.linear_model import LogisticRegression #It's a binary classification problem (malignant vs. benign)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay)  #Performamce Metrices
import matplotlib.pyplot as plt  #For Data Visualisation
import pickle #Convert Target Values in Binary Categories

df['diagnosis']=df['diagnosis'].map({'M':0,'B':1}) #Converts M=0 & B=1
x=df.drop('diagnosis',axis=1)   #Features
y=df['diagnosis']  #Target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)  #Split training and testing data

model=LogisticRegression(max_iter=500)        #Load Model
model.fit(x_train,y_train)        #train model on training data

y_pred=model.predict(x_test)    #Predict on testing data

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

import pickle
pickle.dump(model, open("model.pkl", "wb"))   #to save model

