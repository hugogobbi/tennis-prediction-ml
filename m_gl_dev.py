# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 22:01:58 2020

@author: hgobb
"""
import numpy as np
import pandas as pd
# from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.ensemble import HistGradientBoostingClassifier1
from sklearn import svm
from sklearn import neighbors
from sklearn import neural_network
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import statistics 
import scipy.stats as st
import random
from pickle import dump
# import xgboost as xgb

from sklearn.model_selection import cross_val_score
# import pandas_profiling
# sys.setrecursionlimit(10000)
pd.set_option('display.max_columns', 500)


#Code variables
inputfilepath =  r'C:\Users\hgobb\Documents\Aleph\tennis_data\m_gl_data.csv'
scaler_path = r'C:\Users\hgobb\Documents\Aleph\scaler/'

#   parameters
# year list 
yearlist = ['2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024']
# yearlist = ['2019']
max_rank_to_filter = 200
# regenerate pickles('y') or pickup saved ones ---to add this feature
rescale = 'y'
# features y/n Improve this to a dictionary in order to make it possible to loop
f_court='y'
f_surface='y'
f_p1_rank='y'
f_p2_rank='y'
f_p1_age='y'
f_p1_weight='n'
f_p1_height='y'
f_p1_hand='y'
f_p1_backhand='y'
f_p2_age='y'
f_p2_weight='n'
f_p2_height='y'
f_p2_hand='y'
f_p2_backhand='y'
f_p1_fatigue='y'
f_p2_fatigue='y'
f_p1_b365='y'
f_p2_b365='n'
f_p1_height_sqr='n'
f_p2_height_sqr='n'
f_p1_age_sqr='n'
f_p2_age_sqr='n'
# features to rework
# belo list probably not used
f_standardize = ['p1_rank','p2_rank','p1_age','p1_height','p2_age','p2_weight','p2_height','fatigue_p1','fatigue_p2','p1_b365','p2_b365']
f_dummy = ['court','p1_hand','p1_backhand','p2_hand','p2_backhand']
f_polynomial = ['p1_age','p1_height','p2_age','p2_height']

f_poly_dic =	{'p1_age':[f_p1_age_sqr,'p1_age_sqr'] ,'p1_height':[f_p1_height_sqr,'p1_height_sqr'],'p2_age':[f_p2_age_sqr,'p2_age_sqr'],'p2_height':[f_p2_height_sqr,'p2_height_sqr']}
f_standardize_dic = {'p1_rank':f_p1_rank ,'p2_rank': f_p2_rank,'p1_age': f_p1_age,'p1_height': f_p1_height,'p2_age': f_p2_age,'p2_weight': f_p2_weight,'p2_height': f_p2_height,'p1_fatigue': f_p1_fatigue,'p2_fatigue': f_p2_fatigue,'p1_b365': f_p1_b365,'p2_b365': f_p2_b365,'p1_age_sqr' : f_p1_age_sqr,'p1_height_sqr' : f_p1_height_sqr,'p2_age_sqr':f_p2_age_sqr,'p2_height_sqr':f_p2_height_sqr}       

################################START####################################
featurelist = []
# Loading the data
res_data = pd.read_csv(inputfilepath)
df_res= res_data.copy()

# filter de df with the year list
df_res = df_res[df_res['e_date'].astype(str).str[:4].isin(yearlist)]
# filter max ranking
df_res = df_res[df_res['p1_rank']<max_rank_to_filter]
df_res = df_res[df_res['p2_rank']<max_rank_to_filter]

# Preparing the features

# dummy valid variables
hand_v = ['Left-Handed','Right-Handed']
backhand_v = ['Two-Handed Backhand','One-Handed Backhand']
court_v = ['Indoor', 'Outdoor']

if f_court == 'y':
    df_res = df_res[df_res['court'].astype(str).isin(court_v)]
    df_res['outdoor'] = np.where(df_res['court']=='Outdoor', 1, 0) 
    featurelist.append('outdoor')

if f_p1_hand == 'y':
    df_res = df_res[df_res['p1_hand'].astype(str).isin(hand_v)]
    df_res['p1_right'] = np.where(df_res['p1_hand']=='Right-Handed', 1, 0) 
    featurelist.append('p1_right')

if f_p1_backhand == 'y':
    df_res = df_res[df_res['p1_backhand'].astype(str).isin(backhand_v)]
    df_res['p1_oneback'] = np.where(df_res['p1_backhand']=='One-Handed Backhand', 1, 0) 
    featurelist.append('p1_oneback')

if f_p2_hand == 'y':
    df_res = df_res[df_res['p2_hand'].astype(str).isin(hand_v)]
    df_res['p2_right'] = np.where(df_res['p2_hand']=='Right-Handed', 1, 0) 
    featurelist.append('p2_right')

if f_p2_backhand == 'y':
    df_res = df_res[df_res['p2_backhand'].astype(str).isin(backhand_v)]
    df_res['p2_oneback'] = np.where(df_res['p2_backhand']=='One-Handed Backhand', 1, 0) 
    featurelist.append('p2_oneback')


#one hot encoder surface 
if f_surface == 'y':
    cat_encoder = OneHotEncoder()
    surf_cat = df_res[['surface']]
    surf_cat_1hot = cat_encoder.fit_transform(surf_cat)
    surf_cat_1hot = surf_cat_1hot.toarray()
    a = cat_encoder.categories_
    surf_names = list(a[0])
    surf_names = [x.lower() for x in surf_names]
    surf_df = pd.DataFrame(data=surf_cat_1hot, index=df_res.index, columns=surf_names)
    surf_df =surf_df.drop(surf_names[len(surf_names)-1],axis = 1 )
    df_res= pd.concat([df_res, surf_df], axis=1, sort=False)
    featurelist.extend(surf_names[:len(surf_names)-1])


# squared features
for key in f_poly_dic:
    if f_poly_dic[key][0] == 'y':
        df_res = df_res[df_res[key]>0]
        df_res[f_poly_dic[key][1]] = df_res[key]**2
        featurelist.append(f_poly_dic[key][1]+'_std')

# standardize and filter numerical features
for akey in f_standardize_dic:
    if f_standardize_dic[akey] == 'y':
        if 'fatigue' not in akey:
            df_res = df_res[df_res[akey]>0]
        if rescale =='y':
            scaler = StandardScaler()
            scaler.fit(df_res[[akey]])
            df_res[akey+'_std'] = scaler.fit_transform(df_res[[akey]])
            dump(scaler, open(scaler_path + akey +'.pkl', 'wb'))
        else:
            print('to complete ths bit of code')
        if 'sqr' not in akey :
            featurelist.append(akey+'_std')

#shuffle keep same games together.
df_res['copy_index'] = df_res.index
shuffle_res = list(set(df_res['result_id'].tolist()))
random.shuffle(shuffle_res)
shuffler = pd.DataFrame( shuffle_res, columns = ['result_id_shuffled'])
df_res1 = shuffler.merge(df_res,how = 'inner', left_on = 'result_id_shuffled',right_on = 'result_id')[df_res.columns]
df_res1 = df_res1.set_index('copy_index')

#  checks
# df_res.to_csv(r'C:\Users\hgobb\Documents\Aleph\tennis_data\check.csv')

#Sets definition
df_res1 = df_res1[featurelist + ['is_winner']]
r_train = 0.8
r_validation = 0.10
train, validate, test = np.split(df_res1, [int(r_train*len(df_res1)), int(((r_validation+r_train))*len(df_res1))])
X , y= np.array(train[featurelist]) ,np.array(train[['is_winner']])
X_val , y_val= np.array(validate[featurelist]) ,np.array(validate[['is_winner']])
X_test , y_test= np.array(test[featurelist]) ,np.array(test[['is_winner']])



# broker score b365

df_score = df_res[['result_id','e_date','p1_rank', 'p2_rank','p1_b365', 'p2_b365','is_winner']]
df_score = df_score[(df_score['p1_b365']>0.5) & (df_score['p2_b365']>0.5) ]
df_score['was_favorite'] = np.where((df_score['p1_b365'] <= df_score['p2_b365'] ), 1, 0)
accuracy_score(np.array(df_score[['is_winner']]),np.array(df_score[['was_favorite']]))
precision_score(np.array(df_score[['is_winner']]),np.array(df_score[['was_favorite']]))
accuracy_broker = df_score[(df_score['was_favorite']) == (df_score['is_winner']==1)].shape[0] / df_score.shape[0]
return_fav_strat = (sum(df_score['is_winner']*df_score['was_favorite']*df_score['p1_b365'])/df_score[df_score['was_favorite']==1].shape[0]) -1
return_chall_strat = (sum(df_score['is_winner']*(-1)*(df_score['was_favorite']-1)*df_score['p1_b365'])/df_score[df_score['was_favorite']==0].shape[0]) -1
print('accuracy broker: ' + str(round(accuracy_broker*100,2)) + '%')
print(('B365 Favourite strat return: ' + str(round(return_fav_strat*100,2)) + '%'))
print(('B365 Challenger strat return: ' + str(round(return_chall_strat*100,2)) + '%'))

 

#### Logistic regression
print('Logistic regression')
clf = LogisticRegression(random_state=0).fit(X, y.ravel())
clf.predict(X[:5, :])
y[:5]
clf.predict_proba(X[:5, :])
print(clf.score(X, y))
print(clf.score(X_val, y_val))
print(clf.score(X_test, y_test))


clf_etc0 = ensemble.ExtraTreesClassifier(n_estimators=400, criterion ='entropy', random_state=0,max_features='auto')
clf_etc0.fit(X, y.ravel())
print('Extra trees classifier')
print(clf_etc0.score(X, y))
print(clf_etc0.score(X_val, y_val))
print(clf_etc0.score(X_test, y_test))
# dump(clf_etc0, open(r'C:\Users\hgobb\Documents\Aleph\Model_1\clf_etc0.pkl', 'wb'))


clf_etc0.predict_proba(X_test[:1,:])
clf_etc0.predict(X_test[:5,:])
feature_importance  = clf_etc0.feature_importances_
feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        clf_etc0.estimators_], 
                                        axis = 0)  
plt.bar(featurelist, feature_importance_normalized) 
plt.xlabel('Feature Labels') 
plt.xticks(rotation=90)
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show() 

# clf_etc0.predict_proba(X_val[:5, :])


###clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf_etc = ensemble.ExtraTreesClassifier(n_estimators=400, random_state=1,max_features='auto')
clf_etc.fit(X, y.ravel())
print('Extra trees classifier')
print(clf_etc.score(X, y))
print(clf_etc.score(X_val, y_val))
print(clf_etc.score(X_test, y_test))

# #### K-nearest
# print('K-nearest')
# clf_kn = neighbors.KNeighborsClassifier(n_neighbors = 5)
# clf_kn.fit(X, y.ravel())
# print(clf_kn.score(X, y))
# print(clf_kn.score(X_val, y_val))
# print(clf_kn.score(X_test, y_test))
# clf_kn.predict_proba(X[:5, :])
# # #### SVM
# # print('SVM')
# # clf_svc = svm.SVC()
# # clf_svc.fit(X, y.ravel())
# # print(clf_svc.score(X, y))
# # print(clf_svc.score(X_val, y_val))
# # print(clf_svc.score(X_test, y_test))
# #### Neural Networks Better to use keras
# print('Neural Networks')
# clf_nn = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(feature_names)+1, 6), random_state=1,max_iter = 10000)
# clf_nn.fit(X, y.ravel())
# print(clf_nn.score(X, y))
# print(clf_nn.score(X_val, y_val))
# print(clf_nn.score(X_test, y_test))
# #### Random forest try grandient boosted trees  import GradientBoostingClassifier
# #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier
# clf_gbc = ensemble.GradientBoostingClassifier(n_estimators=100)
# clf_gbc.fit(X, y.ravel())
# print('Gradient Boosting Classifier')
# print(clf_gbc.score(X, y))
# print(clf_gbc.score(X_val, y_val))
# print(clf_gbc.score(X_test, y_test))



clf_rf = ensemble.RandomForestClassifier(max_depth=60, random_state=1)
clf_rf.fit(X, y.ravel())
print('Random forest')
print(clf_rf.score(X, y))
print(clf_rf.score(X_val, y_val))
print(clf_rf.score(X_test, y_test))

clf_rf0 = ensemble.RandomForestClassifier()
clf_rf0.fit(X, y.ravel())
print('Random forest')
print(clf_rf0.score(X, y))
print(clf_rf0.score(X_val, y_val))
print(clf_rf0.score(X_test, y_test))
# clf_rf0.predict_proba(X[:5, :])


###clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf_etc = ensemble.ExtraTreesClassifier(n_estimators=200, random_state=1,max_features='auto')
clf_etc.fit(X, y.ravel())
print('Extra trees classifier')
print(clf_etc.score(X, y))
print(clf_etc.score(X_val, y_val))
print(clf_etc.score(X_test, y_test))



#########Ada
clf_ada = ensemble.AdaBoostClassifier()
clf_ada.fit(X, y.ravel())
print('Ada Boost')
print(clf_ada.score(X, y))
print(clf_ada.score(X_val, y_val))
print(clf_ada.score(X_test, y_test))



models = []
models.append(("LogisticRegression",LogisticRegression(max_iter=1000)))
models.append(("SVC",SVC()))
models.append(("LinearSVC",LinearSVC()))
models.append(("KNeighbors",KNeighborsClassifier()))
models.append(("DecisionTree",DecisionTreeClassifier()))
models.append(("RandomForest",RandomForestClassifier()))
rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                max_depth=10, random_state=0, max_features=None)
models.append(("RandomForest2",rf2))
models.append(("MLPClassifier",MLPClassifier(solver='lbfgs', random_state=0,max_iter=1000)))


# from https://www.kaggle.com/code/tflare/testing-multiple-models-with-scikit-learn-0-79425/notebook

results = []
names = []
for name,model in models[8]:
    print(name )
    result = cross_val_score(model, X, y.ravel(),  cv=3)
    names.append(name)
    results.append(result)


for i in range(len(names)):
    print(names[i],results[i].mean())