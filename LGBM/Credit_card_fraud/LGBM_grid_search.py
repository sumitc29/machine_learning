"""this is paramenter searching stage where we will find the parameters using grid search"""


from sklearn.model_selection import train_test_split, GridSearchCV



train=train_backup
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$created training and testing files$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$satrted on trainng data$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



"""handling missing and getting binary columns to be used from missing *"""

"""creating new column for each existing column having null value with binary """
temp_train=pd.DataFrame()
for each in train.columns:
  #temp_train[each]=train[each]
  if (train[each].isnull().sum() != 0):
    str1=each+"_null"
    temp_train[str1]=train[each].isnull()*1
    
temp_train['isFraud']=train['isFraud']
null_corr=temp_train.corr()


"""columns to be kept in final dataframe for prediction from th temp_train dataframe"""

null_col1=null_corr['isFraud'][(null_corr['isFraud']<-0.15)].index.to_list()
null_col2=null_corr['isFraud'][(null_corr['isFraud']>0.15)].index.to_list()

"""removing isFraud from  null_col2 since it is useless for this"""
null_col2=null_col2[:-1]

print(train.shape)

"""Adding this column to the original dataframes"""
for each in null_col1:
  train[each]=temp_train[each]

for each in null_col2:
  train[each]=temp_train[each]
  
  
print(train.shape)



"""removing pandas column having more han 85% of missing"""

high_missing=[each for each in train.columns if train[each].isnull().sum()/len(train)>0.85]
len(high_missing)
print(train.shape)
train=train.drop(high_missing,axis=1)
print(train.shape)


"""imputing missing values with mode for categorical values variable"""
cat_col=[each for each in train.columns if (train[each]).dtype=='O']

def imp_mode(a):
  mod=a.mode()
  a.fillna(mod[0],inplace= True)
  return a
  
  
train[cat_col]=train[cat_col].apply(imp_mode)


"""imputing missing value for discrete of Numeric values"""
disct_col=[col for col in train.columns if (train[col]).dtype != 'O' and train[col].nunique()  < 200]

train[disct_col]=train[disct_col].apply(imp_mode)

"""imputing missing values for continuouws variable of numeric"""
cont_col=[col for col in train.columns if (train[col]).dtype != 'O' and train[col].nunique()  >= 200]

def imp_median(a):
  med=a.median()
  a.fillna(med,inplace= True)
  return a

train[cont_col]=train[cont_col].apply(imp_median)



"""try 1 more imputation with using new misssing(for categorical)  and similar for others"""


"""label encoding to make all the categorical to numerical"""
from sklearn.preprocessing import LabelEncoder
label_encoder=[]
for each in cat_col:
  
  lb=LabelEncoder()
  train[each]=lb.fit_transform(train[each])
  label_encoder.append(lb)
  
print([col for col in train.columns if train[col].dtype == 'O'])


"""getting correlation and findig most correlated columns out of all"""
temp_corr=(train.corr()['isFraud'])
to_keep=[each for each in temp_corr.index if temp_corr[each] > 0.05 or temp_corr[each] < -0.05]
print(train.shape)
train=train[to_keep]
print(train.shape)


"""creating training and testing dataset"""
y=train['isFraud']
train=train.drop(['isFraud'],axis=1)



print('normalizing data')
#normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train=scaler.fit_transform(train)


train=pd.DataFrame(train)
"""NN algorithm"""


"""using LGBM to create model."""

import lightgbm as lgb

"""d_train = lgb.Dataset(train, label=y)


params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 2**8
params['max_depth'] = -1

params['n_estimators']=2**9
params['max_bin']=255

clf = lgb.train(params, d_train, 100)"""













params = {'boosting_type': 'gbdt',
          
          'objective': 'binary',
          'nthread': 4, # to get cpu speed should be equal to number of processors
          'learning_rate': 0.05,
          'max_bin': 255,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 2,
          'reg_lambda': 2,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'auc',
          'random_state': 21}



# Create parameters to search
gridParams = {
    'learning_rate': [0.005,0.01,0.05,0.1,0.5],
    'max_depth' : [3,5, 9,12 ],
    'n_estimators': [2**3,2**5,2**7,2**9],
    'num_leaves': [2**3,2**4,2**5,2**7],
    'boosting_type' : ['gbdt','dart'],
    'colsample_bytree' : [0.65, 0.7,0.9,1],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2,2,5],
    'reg_lambda' : [1,1.2,1.4,2,5],
    "max_bin" : [100,255,512]
    }

mdl = lgb.LGBMClassifier(params)
print(mdl.get_params().keys())


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.5, random_state=42, stratify = y)

grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)
# Run the grid
grid.fit(X_train, y_train)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)








