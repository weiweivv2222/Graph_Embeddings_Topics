# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:00:15 2019

@author: xg16060
"""
# data preparation packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

#modeling packages 
from sklearn import tree
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score

#from sklearn.naive_bayes import GaussianNB
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel, f_regression

#tuning parameters
import random
from sklearn import model_selection



#%% functions

def convert_cate_num_freq(df):
    '''
    the value of the one hot encode equals to the Frequency levels.
    '''        
    ohe = pd.DataFrame(index=df.index)
    for col in df:
        dummies = pd.get_dummies(df[col], prefix=col)
        ohe = pd.concat([ohe, dummies.div(dummies.shape[1])], axis=1)
    return ohe

def normalization(df):
    '''
    for the numeric features : normalize all values [ 0 , 1]
    '''    
    # print(df.drop(['Unnamed: 0'],axis=1).transpose())
    min_max_scaler = preprocessing.MinMaxScaler()
    df_normalized = pd.DataFrame(min_max_scaler.fit_transform(df))
    
    #add columns and index to the normalizaed table
    df_normalized.columns = df.columns
    df_normalized.index = df.index
    return df_normalized


def get_numeric_df(df):
    '''
    For the numerical features, we normalized them . 
    For the categorical features, we convert them to the numerical features and assigne the frequency value to each feature.
    '''     
    #select the numerical features
    df_continuous_normalized = df.select_dtypes(include=['number'])
   
    if not df_continuous_normalized.empty: df_continuous_normalized = normalization(df_continuous_normalized)
    
    #select the categorical features, be aware of the data types in your original data types. 
    df_categorical_hotencoded = df.select_dtypes(include=['object', 'category'])# customerized
    
    # call the convert_cate_num_freq function for convert the categorical features into the numerical features and get the frequence values 
    if not df_categorical_hotencoded.empty: df_categorical_hotencoded = convert_cate_num_freq(df_categorical_hotencoded)
    
    return pd.concat([df_continuous_normalized,df_categorical_hotencoded], axis=1)


def display_ROC_curve(y_test, y_pred):
    '''
    plot ROC curve 
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def evaluatePred(y_test, y_pred, optimal_cutoff):
    '''
    Evaluation metrics (# get accuracy the confusion matrix and ROC of the test dataset)
    '''
    y_pred = y_pred[:,0]
# TODO: be careful that false positive and false negative might be inversed on the powerpoint (similar resuls)
#     optimal cutoff is only optimised for logistic regression ?
    print('The default cutoff')
    y_pred_cutoff_05 = np.where(y_pred > 0.5, 0, 1)
    KPI(y_test, y_pred_cutoff_05)
    if optimal_cutoff != 0.5:
        print('')
        print('The optimal cutoff: ')
        y_pred_cutoff_optimal = np.where(y_pred > optimal_cutoff, 0, 1)
        KPI(y_test, y_pred_cutoff_optimal)
    print('')

#data preprocessing functions
    
def removeMissingProvince(df):
    '''
    # function to delete the customers without any known location, and delete the location string feature
    '''
    df = df[df['PROVINCIE'] != 'Onbekend'].copy()
    df.drop('PROVINCIE', axis=1, inplace=True)
    return df

# results quality analysis functions
def f_importances(coef, names, importance_treshold):
    '''
    plot the feature importance 
    '''
    imp = coef
    # imp,names = zip(*sorted(zip(imp,names))[-10:])
    imp,names = zip(*sorted(i for i in zip(imp,names) if (i[0] >= importance_treshold or i[0] <= -importance_treshold)))
    plt.rcParams["font.size"] = 8
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(right=0.99)
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

def f_importances_abs(coef, names, importance_treshold):
    imp = abs(coef)
    # imp,names = zip(*sorted(zip(imp,names))[-10:])
    imp,names = zip(*sorted(i for i in zip(imp,names) if i[0] >= importance_treshold))
    plt.rcParams["font.size"] = 8
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(right=0.99)
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

def KPI(y_test, y_pred):
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('accuracy score: ', accuracy_score(y_test, y_pred))
    print('precision score: ', precision_score(y_test, y_pred))
    print('recall score: ', recall_score(y_test, y_pred))
    print('f1 score: ', f1_score(y_test, y_pred))
    print('the value of the AUC: ', roc_auc_score(y_test, y_pred))
    print('the value of the MAE: ', mean_absolute_error(y_test, y_pred))    #should be used on probabilities, or non binary
    print('the value of the MSE: ', mean_squared_error(y_test, y_pred))     #should be used on probabilities, or non binary
    print('the value of the R2: ', r2_score(y_test, y_pred))
    display_ROC_curve(y_test, y_pred)


def compare_algorithms(X_train, y_train, X_test, y_test, optimal_cutoff):

    ##model: naive bayes
    print('NB')
    print(y_test.value_counts()[y_train.value_counts()[:1].index.tolist()[0]]/sum(y_test.value_counts().tolist()))

    ## model: k-nearest-neighbors
    print('KNN')
    n_neighbors_values = {'n_neighbors': range(4, 6)}
    KNNclassifier_PK = KNeighborsClassifier()
    model_KNN_PK = model_selection.GridSearchCV(KNNclassifier_PK, param_grid=n_neighbors_values, scoring='accuracy').fit(X_train, y_train_pk_balanced)
    y_pred_KNN = model_KNN_PK.predict_proba(X_test)
    evaluatePred(y_test, y_pred_KNN, optimal_cutoff)

    ## model: logitic regression(print features coefficients + stats)
    print('LR')
# TODO: verify cross-validation LR from sklearn library
    # model_LR = LogisticRegressionCV(random_state=123, solver='liblinear').fit(X_train, y_train)
    model_LR = LogisticRegression(random_state=123, solver='liblinear').fit(X_train, y_train)
    # model_crossValidation_mean_score = cross_val_score(model_LR, X_train, y_train, cv=10).mean()
    # print("model crossValidation mean score", model_crossValidation_mean_score)
    y_pred_LR = model_LR.predict_proba(X_test)
    evaluatePred(y_test, y_pred_LR, optimal_cutoff)
    f_importances_abs(np.array(model_LR.coef_[0]), X_test.columns, 1)

    ## model: decision tree (
    print('Tree')
    model_treeClassifier = tree.DecisionTreeClassifier().fit(X_train, y_train)
    y_pred_treeClassifier = model_treeClassifier.predict_proba(X_test)
    evaluatePred(y_test, y_pred_treeClassifier, optimal_cutoff)

    ##model: random forest print feature importance)
    print('RF')
    model_RF = RandomForestClassifier().fit(X_train, y_train)
    y_pred_RF = model_RF.predict_proba(X_test)
    evaluatePred(y_test, y_pred_RF, optimal_cutoff)

    ##model: xgBoost
    print('XGBoost')
    model_XGB = xgb.XGBClassifier().fit(X_train, y_train)
    y_pred_XGB = model_XGB.predict_proba(X_test)
    evaluatePred(y_test, y_pred_XGB, optimal_cutoff)

    ##model: SVM
    print('SVM')
    model_SVM = SVC(kernel='linear', probability=True).fit(X_train, y_train)
    y_pred_SVM = model_SVM.predict_proba(X_test)
    evaluatePred(y_test, y_pred_SVM, optimal_cutoff)
    f_importances_abs(np.array(model_SVM.coef_[0]), X_test.columns)



#%%load the data

# load_location="C:\\Users\\xg16060\\OneDrive - APG\\tanguy table\\"
# save_location = "C:\\Users\\xg16060\\OneDrive - APG\\tanguy table\\constructData\\"
load_location = "C:\\Users\\xg16137\\PycharmProjects\\NormalizationOneHot\\Data\\"
save_location = "C:\\Users\\xg16137\\PycharmProjects\\NormalizationOneHot\\ModifiedData\\"

# the three different embeddings of the feature location
dfTopLocCustEmb = pd.read_csv(load_location + "GLoc_topic_Loc_cust_embeddings.csv")
dfTopLocEmb = pd.read_csv(load_location + "GLocation_topic_Location_embeddings.csv")
dfprovinceEmb = pd.read_csv(load_location + "Gprovince_embeddings.csv")

# the balanced training dataset
dfTrain_balanced_rose_pk = pd.read_csv(load_location + "training_balanced_rose_pk.csv", encoding ="ISO-8859-1")
dfTrain_balanced_rose_dz = pd.read_csv(load_location + "training_balanced_rose_dz.csv", encoding ="ISO-8859-1")

# the imbalance traning data
dfTrain_imbalanced = pd.read_csv(load_location + "training_imbalance.csv", encoding ="ISO-8859-1")

# the test dataset
dfTest_dataset = pd.read_csv(load_location + "testing.csv", encoding ="ISO-8859-1")

#%% build the BDA table 

#data preparation     

# transpose the embedding features
dfTopLocCust_normOneHot = dfTopLocCustEmb.drop(['Unnamed: 0'],axis=1).transpose()
dfTopLocEmb_normOneHot = dfTopLocEmb.drop(['Unnamed: 0'],axis=1).transpose()
dfprovinceEmb_normOneHot = dfprovinceEmb.drop(['Unnamed: 0'],axis=1).transpose()

# add the "province" feature
dfTopLocCust_normOneHot['PROVINCIE'] = dfTopLocCust_normOneHot.index
dfTopLocEmb_normOneHot['PROVINCIE'] = dfTopLocEmb_normOneHot.index
dfprovinceEmb_normOneHot['PROVINCIE'] = dfprovinceEmb_normOneHot.index

###################################################################################################################################################################
#%%

# add the location embeddings to the costumer data, via the key: 'PROVINCIE'.

# get the complete balanced training datasets for the PK topic
df_CustData_TopLocCust_included_pk = pd.merge(dfTrain_balanced_rose_pk, dfTopLocCust_normOneHot, how='left', on='PROVINCIE')
df_CustData_TopLocEmb_included_pk = pd.merge(dfTrain_balanced_rose_pk, dfTopLocEmb_normOneHot, how='left', on='PROVINCIE')
df_CustData_provinceEmb_included_pk = pd.merge(dfTrain_balanced_rose_pk, dfprovinceEmb_normOneHot, how='left', on='PROVINCIE')

# get the complete balanced training datasets for the DZ topic
df_CustData_TopLocCust_included_dz = pd.merge(dfTrain_balanced_rose_dz, dfTopLocCust_normOneHot, how='left', on='PROVINCIE')
df_CustData_TopLocEmb_included_dz = pd.merge(dfTrain_balanced_rose_dz, dfTopLocEmb_normOneHot, how='left', on='PROVINCIE')
df_CustData_provinceEmb_included_dz = pd.merge(dfTrain_balanced_rose_dz, dfprovinceEmb_normOneHot, how='left', on='PROVINCIE')

# get the complete imbalanced training datasets
df_CustData_TopLocCust_included_imbalanced = pd.merge(dfTrain_imbalanced, dfTopLocCust_normOneHot, how='left', on='PROVINCIE')
df_CustData_TopLocEmb_included_imbalanced = pd.merge(dfTrain_imbalanced, dfTopLocEmb_normOneHot, how='left', on='PROVINCIE')
df_CustData_provinceEmb_included_imbalanced = pd.merge(dfTrain_imbalanced, dfprovinceEmb_normOneHot, how='left', on='PROVINCIE')

# get the test dataset
df_CustData_Test_TopLocCust_included = pd.merge(dfTest_dataset, dfTopLocCust_normOneHot, how='left', on='PROVINCIE')
df_CustData_Test_TopLocEmb_included = pd.merge(dfTest_dataset, dfTopLocEmb_normOneHot, how='left', on='PROVINCIE')
df_CustData_Test_provinceEmb_included = pd.merge(dfTest_dataset, dfprovinceEmb_normOneHot, how='left', on='PROVINCIE')


###################################################################################################################################################################
#%%
# remove the clients without the location

# PK
df_CustData_TopLocCust_included_pk = removeMissingProvince(df_CustData_TopLocCust_included_pk)
df_CustData_TopLocEmb_included_pk = removeMissingProvince(df_CustData_TopLocEmb_included_pk)
df_CustData_provinceEmb_included_pk = removeMissingProvince(df_CustData_provinceEmb_included_pk)

# DZ
df_CustData_TopLocCust_included_dz = removeMissingProvince(df_CustData_TopLocCust_included_dz)
df_CustData_TopLocEmb_included_dz = removeMissingProvince(df_CustData_TopLocEmb_included_dz)
df_CustData_provinceEmb_included_dz = removeMissingProvince(df_CustData_provinceEmb_included_dz)

# imbalanced
df_CustData_TopLocCust_included_imbalanced = removeMissingProvince(df_CustData_TopLocCust_included_imbalanced)
df_CustData_TopLocEmb_included_imbalanced = removeMissingProvince(df_CustData_TopLocEmb_included_imbalanced)
df_CustData_provinceEmb_included_imbalanced = removeMissingProvince(df_CustData_provinceEmb_included_imbalanced)

# for test
df_CustData_Test_TopLocCust_included = removeMissingProvince(df_CustData_Test_TopLocCust_included)
df_CustData_Test_TopLocEmb_included = removeMissingProvince(df_CustData_Test_TopLocEmb_included)
df_CustData_Test_provinceEmb_included = removeMissingProvince(df_CustData_Test_provinceEmb_included)

###################################################################################################################################################################

#%% the selected features from R code

selected_features_RandomForest_pk = ["LEEFT", "INCOME_PARTTIME_LATEST_JOB", "PARTTIME_INCOME_TOT", "COMMUNICATIONCHOICE", "TYPE_DLN_BIGGEST_JOB"]
selected_features_RandomForest_dz = ["LEEFT", "OMS_SMLVRM", "SECTOR_MMS_LATEST_JOB", "OMS_GESLACHT", "SECTOR_MMS_BIGGEST_JOB", "INCOME_PARTTIME_LATEST_JOB"]
selected_features_Boruta_pk = ["OMS_GESLACHT", "LEEFT", "COMMUNICATIONCHOICE", "NUMBER_OF_DIVORCES", "PARTTIME_INCOME_TOT", "PARTTIME_FACTOR_TOT", "TYPE_DLN_BIGGEST_JOB",  "SECTOR_MMS_BIGGEST_JOB", "PARTTIME_FACTOR_LATEST_JOB", "INCOME_PARTTIME_LATEST_JOB"]
selected_features_Boruta_dz = ["OMS_GESLACHT", "LEEFT", "NUMBER_OF_DIVORCES",  "OMS_SMLVRM","NUMBER_OF_DLN", "PARTTIME_INCOME_TOT", "PARTTIME_FACTOR_TOT", "TYPE_DLN_BIGGEST_JOB", "NETTOPENSION_IND", "PARTTIME_FACTOR_LATEST_JOB", "INCOME_PARTTIME_LATEST_JOB"]

#%%
# normalize the numerical features in the training dataset
# The balanced dataset
# Balanced data without the features selection for the topic PK
X_train_TopLocCust_balanced_allFeatures_pk = get_numeric_df(df_CustData_TopLocCust_included_pk.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))
X_train_TopLocEmb_balanced_allFeatures_pk = get_numeric_df(df_CustData_TopLocEmb_included_pk.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))
X_train_provinceEmb_balanced_allFeatures_pk = get_numeric_df(df_CustData_provinceEmb_included_pk.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))

# Balanced data without the features selection for the topic DZ
X_train_TopLocCust_balanced_allFeatures_dz = get_numeric_df(df_CustData_TopLocCust_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))
X_train_TopLocEmb_balanced_allFeatures_dz = get_numeric_df(df_CustData_TopLocEmb_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))
X_train_provinceEmb_balanced_allFeatures_dz = get_numeric_df(df_CustData_provinceEmb_included_dz.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))

# Balanced data with the features selection by random forest for the topic PK
X_train_TopLocCust_balanced_RandomForest_pk = get_numeric_df(df_CustData_TopLocCust_included_pk.loc[:,selected_features_RandomForest_pk])
X_train_TopLocEmb_balanced_RandomForest_pk = get_numeric_df(df_CustData_TopLocEmb_included_pk.loc[:,selected_features_RandomForest_pk])
X_train_provinceEmb_balanced_RandomForest_pk = get_numeric_df(df_CustData_provinceEmb_included_pk.loc[:,selected_features_RandomForest_pk])

# Balanced data with the features selection by random forest for the topic DZ
X_train_TopLocCust_balanced_RandomForest_dz = get_numeric_df(df_CustData_TopLocCust_included_dz.loc[:,selected_features_RandomForest_dz])
X_train_TopLocEmb_balanced_RandomForest_dz = get_numeric_df(df_CustData_TopLocEmb_included_dz.loc[:,selected_features_RandomForest_dz])
X_train_provinceEmb_balanced_RandomForest_dz = get_numeric_df(df_CustData_provinceEmb_included_dz.loc[:,selected_features_RandomForest_dz])

#  Balanced data with the features selection by boruta for the topic PK
X_train_TopLocCust_balanced_Boruta_pk = get_numeric_df(df_CustData_TopLocCust_included_pk.loc[:,selected_features_Boruta_pk])
X_train_TopLocEmb_balanced_Boruta_pk = get_numeric_df(df_CustData_TopLocEmb_included_pk.loc[:,selected_features_Boruta_pk])
X_train_provinceEmb_balanced_Boruta_pk = get_numeric_df(df_CustData_provinceEmb_included_pk.loc[:,selected_features_Boruta_pk])

# Balanced data with the features selection by boruta for the topic DZ
X_train_TopLocCust_balanced_Boruta_dz = get_numeric_df(df_CustData_TopLocCust_included_dz.loc[:,selected_features_Boruta_dz])
X_train_TopLocEmb_balanced_Boruta_dz = get_numeric_df(df_CustData_TopLocEmb_included_dz.loc[:,selected_features_Boruta_dz])
X_train_provinceEmb_balanced_Boruta_dz = get_numeric_df(df_CustData_provinceEmb_included_dz.loc[:,selected_features_Boruta_dz])

#%%
# The imbalanced dataset
# Imbalanced data without the features selection
X_train_TopLocCust_imbalanced_allFeatures = get_numeric_df(df_CustData_TopLocCust_included_imbalanced.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))
X_train_TopLocEmb_imbalanced_allFeatures = get_numeric_df(df_CustData_TopLocEmb_included_imbalanced.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))
X_train_provinceEmb_imbalanced_allFeatures = get_numeric_df(df_CustData_provinceEmb_included_imbalanced.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))

#  Imbalanced data with the features selection with random forest for the topic pk
X_train_TopLocCust_imbalanced_RandomForest_pk = get_numeric_df(df_CustData_TopLocCust_included_imbalanced.loc[:,selected_features_RandomForest_pk])
X_train_TopLocEmb_imbalanced_RandomForest_pk = get_numeric_df(df_CustData_TopLocEmb_included_imbalanced.loc[:,selected_features_RandomForest_pk])
X_train_provinceEmb_imbalanced_RandomForest_pk = get_numeric_df(df_CustData_provinceEmb_included_imbalanced.loc[:,selected_features_RandomForest_pk])

#  Imbalanced data with the features selection with random forest for the topic dz
X_train_TopLocCust_imbalanced_RandomForest_dz = get_numeric_df(df_CustData_TopLocCust_included_imbalanced.loc[:,selected_features_RandomForest_dz])
X_train_TopLocEmb_imbalanced_RandomForest_dz = get_numeric_df(df_CustData_TopLocEmb_included_imbalanced.loc[:,selected_features_RandomForest_dz])
X_train_provinceEmb_imbalanced_RandomForest_dz = get_numeric_df(df_CustData_provinceEmb_included_imbalanced.loc[:,selected_features_RandomForest_dz])

#   Imbalanced data with the features selection with boruta for the topic pk
X_train_TopLocCust_imbalanced_Boruta_pk = get_numeric_df(df_CustData_TopLocCust_included_imbalanced.loc[:,selected_features_Boruta_pk])
X_train_TopLocEmb_imbalanced_Boruta_pk = get_numeric_df(df_CustData_TopLocEmb_included_imbalanced.loc[:,selected_features_Boruta_pk])
X_train_provinceEmb_imbalanced_Boruta_pk = get_numeric_df(df_CustData_provinceEmb_included_imbalanced.loc[:,selected_features_Boruta_pk])

#   Imbalanced data with the features selection with boruta for the topic dz
X_train_TopLocCust_imbalanced_Boruta_dz = get_numeric_df(df_CustData_TopLocCust_included_imbalanced.loc[:,selected_features_Boruta_dz])
X_train_TopLocEmb_imbalanced_Boruta_dz = get_numeric_df(df_CustData_TopLocEmb_included_imbalanced.loc[:,selected_features_Boruta_dz])
X_train_provinceEmb_imbalanced_Boruta_dz = get_numeric_df(df_CustData_provinceEmb_included_imbalanced.loc[:,selected_features_Boruta_dz])

#%%
# for test dataset : no features selection
X_test_TopLocCust_allFeatures = get_numeric_df(df_CustData_Test_TopLocCust_included.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))
X_test_TopLocEmb_allFeatures = get_numeric_df(df_CustData_Test_TopLocEmb_included.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))
X_test_provinceEmb_allFeatures = get_numeric_df(df_CustData_Test_provinceEmb_included.drop(['KLTID', 'Target', 'TargetDZ', 'TargetPK'], axis=1))


#   features selection with random forest
#       pk
X_test_TopLocCust_RandomForest_pk = get_numeric_df(df_CustData_Test_TopLocCust_included.loc[:,selected_features_RandomForest_pk])
X_test_TopLocEmb_RandomForest_pk = get_numeric_df(df_CustData_Test_TopLocEmb_included.loc[:,selected_features_RandomForest_pk])
X_test_provinceEmb_RandomForest_pk = get_numeric_df(df_CustData_Test_provinceEmb_included.loc[:,selected_features_RandomForest_pk])
#       dz
X_test_TopLocCust_RandomForest_dz = get_numeric_df(df_CustData_Test_TopLocCust_included.loc[:,selected_features_RandomForest_dz])
X_test_TopLocEmb_RandomForest_dz = get_numeric_df(df_CustData_Test_TopLocEmb_included.loc[:,selected_features_RandomForest_dz])
X_test_provinceEmb_RandomForest_dz = get_numeric_df(df_CustData_Test_provinceEmb_included.loc[:,selected_features_RandomForest_dz])

#   features selection with boruta
#       pk
X_test_TopLocCust_Boruta_pk = get_numeric_df(df_CustData_Test_TopLocCust_included.loc[:,selected_features_Boruta_pk])
X_test_TopLocEmb_Boruta_pk = get_numeric_df(df_CustData_Test_TopLocEmb_included.loc[:,selected_features_Boruta_pk])
X_test_provinceEmb_Boruta_pk = get_numeric_df(df_CustData_Test_provinceEmb_included.loc[:,selected_features_Boruta_pk])
#       dz
X_test_TopLocCust_Boruta_dz = get_numeric_df(df_CustData_Test_TopLocCust_included.loc[:,selected_features_Boruta_dz])
X_test_TopLocEmb_Boruta_dz = get_numeric_df(df_CustData_Test_TopLocEmb_included.loc[:,selected_features_Boruta_dz])
X_test_provinceEmb_Boruta_dz = get_numeric_df(df_CustData_Test_provinceEmb_included.loc[:,selected_features_Boruta_dz])

#%%
# Output
# for train
#balanced
y_train_pk_balanced = df_CustData_TopLocCust_included_pk['TargetPK']
y_train_dz_balanced = df_CustData_TopLocCust_included_dz['TargetDZ']

#imbalanced
y_train_pk_imbalanced = df_CustData_TopLocCust_included_imbalanced['TargetPK']
y_train_dz_imbalanced = df_CustData_TopLocCust_included_imbalanced['TargetDZ']

# for test
y_test_pk = df_CustData_Test_TopLocCust_included['TargetPK']
y_test_dz = df_CustData_Test_TopLocCust_included['TargetDZ']



#%%apply all algorithms on the different graph embedings

# construct the balanced and imbalanced combinaisons for TopLocCust, for PK
Diff_Embeddings_balanced_TopLocCust_pk = [["balanced all features", X_train_TopLocCust_balanced_allFeatures_pk, X_test_TopLocCust_allFeatures, 0.7710908], ["balanced random forest", X_train_TopLocCust_balanced_RandomForest_pk, X_test_TopLocCust_RandomForest_pk, 0.5], ["balanced boruta", X_train_TopLocCust_balanced_Boruta_pk, X_test_TopLocCust_Boruta_pk, 0.5]]
Diff_Embeddings_imbalanced_TopLocCust_pk = [["imbalanced all features", X_train_TopLocCust_imbalanced_allFeatures, X_test_TopLocCust_allFeatures, 0.5954308], ["imbalanced random forest", X_train_TopLocCust_imbalanced_RandomForest_pk, X_test_TopLocCust_RandomForest_pk, 0.5], ["imbalanced boruta", X_train_TopLocCust_imbalanced_Boruta_pk, X_test_TopLocCust_Boruta_pk, 0.5]]
Diff_Embeddings_TopLocCust_pk = [[Diff_Embeddings_balanced_TopLocCust_pk, y_train_pk_balanced], [Diff_Embeddings_imbalanced_TopLocCust_pk, y_train_pk_imbalanced]]

# construct the balanced and imbalanced combinaisons for TopLocEmb, for PK
Diff_Embeddings_balanced_TopLocEmb_pk = [["balanced all features", X_train_TopLocEmb_balanced_allFeatures_pk, X_test_TopLocEmb_allFeatures, 0.7710908], ["balanced random forest", X_train_TopLocEmb_balanced_RandomForest_pk, X_test_TopLocEmb_RandomForest_pk, 0.5], ["balanced boruta", X_train_TopLocEmb_balanced_Boruta_pk, X_test_TopLocEmb_Boruta_pk, 0.5]]
Diff_Embeddings_imbalanced_TopLocEmb_pk = [["imbalanced all features", X_train_TopLocEmb_imbalanced_allFeatures, X_test_TopLocEmb_allFeatures, 0.5954308], ["imbalanced random forest", X_train_TopLocEmb_imbalanced_RandomForest_pk, X_test_TopLocEmb_RandomForest_pk, 0.5], ["imbalanced boruta", X_train_TopLocEmb_imbalanced_Boruta_pk, X_test_TopLocEmb_Boruta_pk, 0.5]]
Diff_Embeddings_TopLocEmb_pk = [[Diff_Embeddings_balanced_TopLocEmb_pk, y_train_pk_balanced], [Diff_Embeddings_imbalanced_TopLocEmb_pk, y_train_pk_imbalanced]]

# construct the balanced and imbalanced combinaisons for provinceEmb, for PK
Diff_Embeddings_balanced_provinceEmb_pk = [["balanced all features", X_train_provinceEmb_balanced_allFeatures_pk, X_test_provinceEmb_allFeatures, 0.7710908], ["balanced random forest", X_train_provinceEmb_balanced_RandomForest_pk, X_test_provinceEmb_RandomForest_pk, 0.5], ["balanced boruta", X_train_provinceEmb_balanced_Boruta_pk, X_test_provinceEmb_Boruta_pk, 0.5]]
Diff_Embeddings_imbalanced_provinceEmb_pk = [["imbalanced all features", X_train_provinceEmb_imbalanced_allFeatures, X_test_provinceEmb_allFeatures, 0.5954308], ["imbalanced random forest", X_train_provinceEmb_imbalanced_RandomForest_pk, X_test_provinceEmb_RandomForest_pk, 0.5], ["imbalanced boruta", X_train_provinceEmb_imbalanced_Boruta_pk, X_test_provinceEmb_Boruta_pk, 0.5]]
Diff_Embeddings_provinceEmb_pk = [[Diff_Embeddings_balanced_provinceEmb_pk, y_train_pk_balanced], [Diff_Embeddings_imbalanced_provinceEmb_pk, y_train_pk_imbalanced]]

Diff_Embeddings_pk = [Diff_Embeddings_TopLocCust_pk]#, Diff_Embeddings_TopLocEmb_pk, Diff_Embeddings_provinceEmb_pk


# construct the balanced and imbalanced combinaisons for TopLocCust, for DZ
Diff_Embeddings_balanced_TopLocCust_dz = [["balanced all features", X_train_TopLocCust_balanced_allFeatures_dz, X_test_TopLocCust_allFeatures, 0.7710908], ["balanced random forest", X_train_TopLocCust_balanced_RandomForest_dz, X_test_TopLocCust_RandomForest_dz, 0.5], ["balanced boruta", X_train_TopLocCust_balanced_Boruta_dz, X_test_TopLocCust_Boruta_dz, 0.5]]
Diff_Embeddings_imbalanced_TopLocCust_dz = [["imbalanced all features", X_train_TopLocCust_imbalanced_allFeatures, X_test_TopLocCust_allFeatures, 0.5954308], ["imbalanced random forest", X_train_TopLocCust_imbalanced_RandomForest_dz, X_test_TopLocCust_RandomForest_dz, 0.5], ["imbalanced boruta", X_train_TopLocCust_imbalanced_Boruta_dz, X_test_TopLocCust_Boruta_dz, 0.5]]
Diff_Embeddings_TopLocCust_dz = [[Diff_Embeddings_balanced_TopLocCust_dz, y_train_dz_balanced], [Diff_Embeddings_imbalanced_TopLocCust_dz, y_train_dz_imbalanced]]

# construct the balanced and imbalanced combinaisons for TopLocEmb, for DZ
Diff_Embeddings_balanced_TopLocEmb_dz = [["balanced all features", X_train_TopLocEmb_balanced_allFeatures_dz, X_test_TopLocEmb_allFeatures, 0.7710908], ["balanced random forest", X_train_TopLocEmb_balanced_RandomForest_dz, X_test_TopLocEmb_RandomForest_dz, 0.5], ["balanced boruta", X_train_TopLocEmb_balanced_Boruta_dz, X_test_TopLocEmb_Boruta_dz, 0.5]]
Diff_Embeddings_imbalanced_TopLocEmb_dz = [["imbalanced all features", X_train_TopLocEmb_imbalanced_allFeatures, X_test_TopLocEmb_allFeatures, 0.5954308], ["imbalanced random forest", X_train_TopLocEmb_imbalanced_RandomForest_dz, X_test_TopLocEmb_RandomForest_dz, 0.5], ["imbalanced boruta", X_train_TopLocEmb_imbalanced_Boruta_dz, X_test_TopLocEmb_Boruta_dz, 0.5]]
Diff_Embeddings_TopLocEmb_dz = [[Diff_Embeddings_balanced_TopLocEmb_dz, y_train_dz_balanced], [Diff_Embeddings_imbalanced_TopLocEmb_dz, y_train_dz_imbalanced]]

# construct the balanced and imbalanced combinaisons for provinceEmb, for DZ
Diff_Embeddings_balanced_provinceEmb_dz = [["balanced all features", X_train_provinceEmb_balanced_allFeatures_dz, X_test_provinceEmb_allFeatures, 0.7710908], ["balanced random forest", X_train_provinceEmb_balanced_RandomForest_dz, X_test_provinceEmb_RandomForest_dz, 0.5], ["balanced boruta", X_train_provinceEmb_balanced_Boruta_dz, X_test_provinceEmb_Boruta_dz, 0.5]]
Diff_Embeddings_imbalanced_provinceEmb_dz = [["imbalanced all features", X_train_provinceEmb_imbalanced_allFeatures, X_test_provinceEmb_allFeatures, 0.5954308], ["imbalanced random forest", X_train_provinceEmb_imbalanced_RandomForest_dz, X_test_provinceEmb_RandomForest_dz, 0.5], ["imbalanced boruta", X_train_provinceEmb_imbalanced_Boruta_dz, X_test_provinceEmb_Boruta_dz, 0.5]]
Diff_Embeddings_provinceEmb_dz = [[Diff_Embeddings_balanced_provinceEmb_dz, y_train_dz_balanced], [Diff_Embeddings_imbalanced_provinceEmb_dz, y_train_dz_imbalanced]]

Diff_Embeddings_dz = [Diff_Embeddings_TopLocCust_dz, Diff_Embeddings_TopLocEmb_dz, Diff_Embeddings_provinceEmb_dz]

#add the random seed in order to produce the same results
random.seed(123)

# Cycle trough all the combination, regarding the subject PK
for embType_pk in Diff_Embeddings_pk:
    for y_train_balanced_or_imbalanced in embType_pk:
        for emb in y_train_balanced_or_imbalanced[0]:
            print(emb[0])
            compare_algorithms(emb[1], y_train_balanced_or_imbalanced[1], emb[2], y_test_pk, emb[3])

## Cycle trough all the combinaiton, regarding the subject DZ
# for embType_dz in Diff_Embeddings_dz:
#     for y_train_balanced_or_imbalanced in embType_dz:
#         for emb in y_train_balanced_or_imbalanced[0]:
#             print(emb[0])
#             compare_algorithms(emb[1], y_train_balanced_or_imbalanced[1], emb[2], y_test_dz, emb[3])