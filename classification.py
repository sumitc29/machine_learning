##classification example dataset used is Spambase. Basically it is used for nlp purpose

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def drop_feature(data_frame,feature):
    data_frame=data_frame.drop([feature],axis=1)
    return data_frame
    
    
os.listdir('../input') 
data=pd.read_csv('../input/train_data.csv')
test_data=pd.read_csv('../input/test_features.csv')
data.info()
test_data.info()
data.head(2)


target=list(set(data.columns)-set(test_data.columns))[0]
data[target].head(3)
data[target].value_counts().plot(kind='bar')
data.isnull().sum()

'''data EDA and feature extraction'''
temp_data=data
temp_data=drop_feature(temp_data,'Id')
temp_data['ham']=np.where(temp_data['ham']==True,1,0)


temp_data=temp_data.as_matrix()
type(temp_data)
x=temp_data[:,:-1]
y=temp_data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)





'''creating model'''

model = MultinomialNB()
model.fit(X_train,y_train)
print('score of naive base',model.score(X_test,y_test))

from sklearn.ensemble import AdaBoostClassifier
ada_model=AdaBoostClassifier()
ada_model.fit(X_train,y_train)
print('score of naive base',ada_model.score(X_test,y_test))
accuracy_score(X_test,y_test)



from xgboost import XGBClassifier
xgb_model=XGBClassifier()
xgb_model.fit(X_train,y_train)
print('score of naive base',xgb_model.score(X_test,y_test))


