import pandas as pd
data = pd.read_csv('/content/Iris.csv')

data = data.sample(frac=1).reset_index(drop = True)
train = data[:135] ; test  = data[135:]

#using autoviml
#!pip install autoviml
from autoviml.Auto_ViML import Auto_ViML


sample_submission=' '
label = 'Species'
model, features, trainm, testm = Auto_ViML(train, label, test, sample_submission, hyper_param='GS', feature_reduction=True, scoring_parameter='weighted-f1', KMeans_Featurizer=False, Boosting_Flag=True, Binning_Flag=False, Add_Poly=False, Stacking_Flag=False,Imbalanced_Flag=False, verbose=0)
