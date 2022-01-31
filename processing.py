# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:10:09 2020

@author: Alise Danielle Midtfjord

This script works as a package, and contains functions used for loading in and 
preprocessing csv-files into usable dataframes. 

"""

# Package imports 
import pandas as pd
import numpy as np
import configs as cfg
from sklearn.model_selection import train_test_split
import xgboost as xgb


def wind_pros(df):
    '''
    Process wind variables to only get informaiton from the sensor from incoming flight direction
    
    :param Dataframe df: Dataframe containing the wind variables
    :return: The Dataframe with the transformed wind variables
    '''

    df.loc[df['direction'] == 1, 'dir02_in'] = df.loc[df['direction'] == 1, 'dir02_1']
    df.loc[df['direction'] == 1, 'dir10_in'] = df.loc[df['direction'] == 1, 'dir10_1']
    df.loc[df['direction'] == 1, 'max02_in'] = df.loc[df['direction'] == 1, 'max02_1']
    df.loc[df['direction'] == 1, 'max10_in'] = df.loc[df['direction'] == 1, 'max10_1']
    df.loc[df['direction'] == 1, 'spe02_in'] = df.loc[df['direction'] == 1, 'spe02_1']
    df.loc[df['direction'] == 1, 'spe10_in'] = df.loc[df['direction'] == 1, 'spe10_1']
    df.loc[df['direction'] == 1, 'across02_in'] = df.loc[df['direction'] == 1, 'across02_1']
    df.loc[df['direction'] == 1, 'along02_in'] = df.loc[df['direction'] == 1, 'along02_1']
    df.loc[df['direction'] == 1, 'ac_calc02_in'] = df.loc[df['direction'] == 1, 'ac_calc02_1']
    df.loc[df['direction'] == 1, 'al_calc02_in'] = df.loc[df['direction'] == 1, 'al_calc02_1']
    
    df.loc[df['direction'] == 19, 'dir02_in'] = df.loc[df['direction'] == 19, 'dir02_2']
    df.loc[df['direction'] == 19, 'dir10_in'] = df.loc[df['direction'] == 19, 'dir10_2']
    df.loc[df['direction'] == 19, 'max02_in'] = df.loc[df['direction'] == 19, 'max02_2']
    df.loc[df['direction'] == 19, 'max10_in'] = df.loc[df['direction'] == 19, 'max10_2']
    df.loc[df['direction'] == 19, 'spe02_in'] = df.loc[df['direction'] == 19, 'spe02_2']
    df.loc[df['direction'] == 19, 'spe10_in'] = df.loc[df['direction'] == 19, 'spe10_2']
    df.loc[df['direction'] == 19, 'across02_in'] = df.loc[df['direction'] == 19, 'across02_2']
    df.loc[df['direction'] == 19, 'along02_in'] = df.loc[df['direction'] == 19, 'along02_2']
    df.loc[df['direction'] == 19, 'ac_calc02_in'] = df.loc[df['direction'] == 19, 'ac_calc02_2']
    df.loc[df['direction'] == 19, 'al_calc02_in'] = df.loc[df['direction'] == 19, 'al_calc02_2']
    
    df = df.drop(['dir02_1','dir02_2', 'dir10_1','dir10_2','max02_1','max02_2', 'max10_1', 'max10_2',
                  'spe02_1','spe02_2','spe10_1','spe10_2','across02_1','across02_2','along02_1',
                  'along02_2','ac_calc02_1','ac_calc02_2','al_calc02_1','al_calc02_2'], axis = 1)
    
    return(df)

def create_dataframe():
    '''
    Create DataFrame from csv-file according to path in configs
    
    :return: Dataframe
    '''

    df = pd.read_csv(cfg.path, index_col = 0)
    df['key'] = df.index.astype(str) + '-' + df['aerodrome'].astype(str)
    df = df.set_index('key')

        
    return(df)


def data_for_both():
    '''
    Create and process two dataframes (classification and regression) to be 
    used for SHAP explanations (the classification dataframe containts all data instances,
    while the regression dataframe containts only the friction limited landings)
    
    :return: XGBoost dMatrix with test data for classification
    :return: XGBoost dMatrix with test data for regression
    :return: XGBoost dMatrix with train data for classification
    :return: XGBoost dMatrix with train data for regression
    :return: Array with train response for classification
    :return: Array with train response for regression
    :return: Array with test response for classification
    :return: Array with test response for regression
    :return: Dataframe with train explanatory variables for classification
    :return: Dataframe with test explanatory variables for classification
    :return: Dataframe with train explanatory variables for regression
    :return: Dataframe with test explanatory variables for regression
    '''

    df = create_dataframe()
    
    fricLim = cfg.fricLim
    mu = cfg.mu
        
    df.loc[df[fricLim] > 0, fricLim] = 1
    df['slippery'] = df[fricLim]
    df.loc[df[mu] > 0.15, 'slippery'] = 0
        
    df['date'] = df.index
    df['key'] = df.index.astype(str) + '-' + df['aerodrome'].astype(str)
    df = df.set_index('key')
      
    df_notoverlap = df[~df.index.duplicated(keep='first')]
    print('Dropped {}'.format(len(df)-len(df_notoverlap)) + ' overlapping flight landings')
    df_new = df_notoverlap.loc[~(df_notoverlap.mu_allRaw <= 0.05),:]
    print('Dropped {}'.format(len(df_notoverlap)-len(df_new)) + ' NIL landings')
    
    df_new.aerodrome = df_new.aerodrome.astype(int)
    df_new.direction = df_new.direction.astype(int)
    df_new = wind_pros(df_new)
    df_new = df_new.drop('direction', axis = 1)
    X = df_new.drop(cfg.unusable_columns, axis = 1)
    X['tmp_abs'] = X.tmp.abs()
    X['rwy_abs'] = X.rwy.abs()

    
    # Set the response to be fricLim
    y = df_new['slippery']
    
    # Set random state
    state = cfg.state
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = state)
    
    y_train_reg = X_train[mu]
    y_test_reg = X_test[mu]
    y_train_clas = y_train
    y_test_clas = y_test

    y_train_reg = y_train_reg.loc[~(pd.isna(X_train.mu_allRaw))]
    y_test_reg = y_test_reg.loc[~(pd.isna(X_test.mu_allRaw))]
    X_train_reg = X_train.loc[~(pd.isna(X_train.mu_allRaw)),:]
    X_test_reg = X_test.loc[~(pd.isna(X_test.mu_allRaw)),:]
    
    X_train_clas = X_train.drop([mu, 'slippery'], axis = 1)
    X_train_reg = X_train_reg.drop([mu, 'slippery'], axis = 1)
    X_test_clas = X_test.drop([mu,'slippery'], axis = 1)
    X_test_reg = X_test_reg.drop([mu,'slippery'], axis = 1)
    
    label_clas = np.array(y_train_clas) 
    label_test_clas = np.array(y_test_clas)
    dtrain_clas = xgb.DMatrix(X_train_clas, label = label_clas)
    dtest_clas = xgb.DMatrix(X_test_clas, label = label_test_clas)
    
    label_reg = np.array(y_train_reg) 
    label_test_reg = np.array(y_test_reg)
    dtrain_reg = xgb.DMatrix(X_train_reg, label = label_reg)
    dtest_reg = xgb.DMatrix(X_test_reg, label = label_test_reg)
    
    return(dtest_clas,dtest_reg,dtrain_clas,dtrain_reg, y_train_clas,y_train_reg, y_test_clas,y_test_reg,
           X_train_clas,X_test_clas,X_train_reg,X_test_reg)