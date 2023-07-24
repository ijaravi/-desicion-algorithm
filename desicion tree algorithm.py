import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

#load iris 
iris=load_iris()
X,y=iris.data,iris.target
#split the data into training and testing sets 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Scaale featurees using Standard scaler 
scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# here we are doing scalling to gert the equility in thee data

#create a dessicion tree classifer  with ID3 algorithm
clf=DecisionTreeClassifier(random_state=42)

#define hyperparameters and theor possible values for   tuning 
#
param_gird={
             'criterion':['gini','entropy'],
             'max_depth':[None,5,10,15],
             'min_samples_split':[2,5,10],
             'min_samples_leaf':[1,2,4]}
#use GridsearchCv to find besthyperparameters
grid_search=GridSearchCV(clf,param_gird ,cv=5)
grid_search.fit(X_train,y_train) 

#get the best hyper parameters 
best_param=grid_search.best_params_
print("besthyperparmeters:",best_param)

#create thee decision tree classifier  witht he best hyper parameters
best_clf=DecisionTreeClassifier(**best_param,random_state=42)
#train the classifer on the training data
best_clf.fit(X_train,y_train)
# make predictions of the test data 
y_pred=best_clf.predict(X_test)
#  calculate the accuracy of rthe moodel 
accuracy=accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)
#print the classified information 
target_names=iris.target_names
print("Classified report:")
print(classification_report(y_test, y_pred,target_names=target_names))
#visualition
#barchart to show the count of each class int he target variables 
plt.figure(figsize=(6,4))
sns.countplot(  x=y,palette='coolwarm')
plt.xticks(ticks=np.unique(y)  , labels=target_names, rotation=45)
plt.xlabel('class')
plt.ylabel('coutn')
plt.title('Class Distribution')
plt.show()
#confusion heat maP
conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='coolwarm',xticklabels=target_names,yticklabels=target_names)
plt.xlabel('predictive class')
plt.ylabel('true class')
plt.title('Class Distribution')
plt.show()
  



