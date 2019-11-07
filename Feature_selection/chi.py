"""using univariate analysis"""
#this is used only for cat vs cat variable and in encoded form

#more the value of chi2 better the correlation between variables
le = preprocessing.LabelEncoder()
y = le.fit_transform(train['target'])

le = preprocessing.LabelEncoder()
x= le.fit_transform(train['independent'])

x = x.reshape(-1,1)
print(chi2(x,y))



"""multivariate analysis"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

le = preprocessing.LabelEncoder()
y = le.fit_transform(train['target'])

x=train[['source_name' , 'destination_name' , 'current_day']]
for each in x.columns:
  le = preprocessing.LabelEncoder()
  x[each] = le.fit_transform(x[each])
   
print(x.shape , y.shape)
chi2_selector = SelectKBest(chi2, k=1)
X_kbest = chi2_selector.fit_transform(x, y)




