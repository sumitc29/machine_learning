'''here data is used from sms-spam-collection-dataset''''
data1=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding = "ISO-8859-1")
data1.info()
for each in data1.columns:
        print(data1[each].head(4))
        
data1.isnull().sum()
data1=data1.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
data1.columns
temp_data1=data1        
        
data1['v1'].value_counts()
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
x = v.fit_transform(temp_data1['v2']) 
x=x.toarray()
x.shape
y=temp_data1['v1']


y=np.where(y=='ham',1,0)
type(y)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)





'''creating model'''

model = MultinomialNB()
model.fit(X_train,y_train)
print('score of naive base',model.score(X_test,y_test))

from sklearn.ensemble import AdaBoostClassifier
ada_model=AdaBoostClassifier()
ada_model.fit(X_train,y_train)
print('score of adaboost',ada_model.score(X_test,y_test))



from xgboost import XGBClassifier
xgb_model=XGBClassifier()
xgb_model.fit(X_train,y_train)
print('score of xgboost',xgb_model.score(X_test,y_test))

'''new method'''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
c = vectorizer.fit_transform(temp_data1['v2'])

print(vectorizer.get_feature_names())
c=c.toarray()
c.shape



C_train, C_test, y_train, y_test = train_test_split(c, y, test_size=0.2, random_state=1)

'''creating model'''

model = MultinomialNB()
model.fit(C_train,y_train)
print('score of naive base',model.score(C_test,y_test))

from sklearn.ensemble import AdaBoostClassifier
ada_model=AdaBoostClassifier()
ada_model.fit(C_train,y_train)
print('score of adaboost',ada_model.score(C_test,y_test))




from xgboost import XGBClassifier
xgb_model=XGBClassifier()
xgb_model.fit(C_train,y_train)
print('score of xgboost',xgb_model.score(C_test,y_test))
