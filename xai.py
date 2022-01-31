# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:52:05 2021

@author: Alise Danielle Midtfjord

This script calculates and displays SHAP values for both the classification
and regression task (either local or global SHAP values)

"""

import shap
import processing as prc
import calculations as calc
import numpy as np
import xgboost as xgb
import pandas as pd
import configs as cfg

# Load data
dtest_clas,dtest_reg,dtrain_clas,dtrain_reg, y_train_clas,y_train_reg, y_test_clas,y_test_reg, X_train_clas,X_test_clas,X_train_reg,X_test_reg = prc.data_for_both()

# Perform an XGBoost classification
bst_clas,y_pred_clas, threshold = calc.classification(dtest_clas,dtrain_clas,y_train_clas, y_test_clas)

# Perform an XGBoost regression
bst_reg,y_pred_reg= calc.regression(dtest_reg,dtrain_reg,y_train_reg, y_test_reg)

# Create new Dataframes and arrays with reset indexes
X_test_index = X_test_reg.copy()
X_test_index['num'] = np.arange(len(X_test_index))
y_test_index = y_test_clas.copy()
y_test_index = y_test_index.reset_index()

# Calculate SHAP values for the classification
explainer_clas = shap.TreeExplainer(bst_clas, data= X_train_clas, check_additivity=False)
shap_values_clas = explainer_clas(X_test_clas)

# Calculate SHAP values for the regression
explainer_reg = shap.TreeExplainer(bst_reg,data = X_train_reg, check_additivity=False)
shap_values_reg = explainer_reg(X_test_reg)

#-----------------------  Local     -----------------------#
# Calcualte local SHAP values for a specific instanse (alter i to decide which instance)

if cfg.shap == 'local':
    
    # Change this
    i = 460
    
    # Print SHAP values with relevant information about the instance
    print('True Slippery: ', y_test_clas.iloc[i])
    if y_pred_clas[i] < threshold:
        print('Pred Prob. Slippery: ', np.round((y_pred_clas[i]/(threshold))*50, decimals = 1), ' %')
    else:
        print('Pred Prob. Slippery: ', np.round((y_pred_clas[i]/(1-threshold))*50+50, decimals = 1), ' %')
    
    data_point_clas = X_test_clas.iloc[i]
    print(' ')
    
    time = data_point_clas.name
    
    if time in X_test_index.index.values:  
        num = X_test_index.loc[time,'num']
        j = num
        y_test_cat = pd.cut(x = pd.Series(y_test_reg[j]), bins = [-1, 0.05,0.075,0.1,0.15,0.2,100],right = True, labels = [0,1,2,3,4,5]) 
        y_pred_cat = pd.cut(x = pd.Series(y_pred_reg[j]), bins = [-1, 0.05,0.075,0.1,0.15,0.2,100],right = True, labels = [0,1,2,3,4,5]) 
        print('True Friction Coefficient: ', np.round(y_test_reg.iloc[j],5))
        print('Pred Friction Coefficient: ', np.round(y_pred_reg[j],5))
        print('True Braking Action: ', y_test_cat[0])
        print('Pred Braking Action: ', y_pred_cat[0])
        shap.plots.waterfall(shap_values_reg[j], max_display=11, show=True)
        
        keys = list(X_train_reg.columns.values)
        values = list(shap_values_reg[j].values)
        numbers = list(shap_values_reg[j].data)
        data_pos = pd.DataFrame(data={'values':values,'numbers':numbers}, index=keys).sort_values(by = "values", ascending=False)[0:5]
        data_neg = pd.DataFrame(data={'values':values,'numbers':numbers}, index=keys).sort_values(by = "values", ascending=True)[0:5]
        
        print(' ')
        print('What makes it slippery:')
        print(' ')
        print(data_neg.loc[:,'numbers'])
        
        print(' ')
        print('What makes it less slippery:')
        print(' ')
        print(data_pos.loc[:,'numbers'])
        
    else:
        new_x = X_test_clas.iloc[i:i+2]
        dtest = xgb.DMatrix(new_x)
        y_new_pred = bst_reg.predict(dtest)
        y_pred_cat = pd.cut(x = pd.Series(y_new_pred[0]), bins = [-1, 0.05,0.075,0.1,0.15,0.2,100],right = True, labels = [0,1,2,3,4,5]) 
        shap_values_reg_ew = explainer_reg(new_x)
        shap.plots.waterfall(shap_values_clas[i], max_display=11, show=True)
        print('Pred Friction Coefficient: ', y_new_pred [0])
        print('Pred Braking Action: ', y_pred_cat[0])
        
        keys = list(X_train_clas.columns.values)
        values = list(shap_values_clas[i].values)
        numbers = list(shap_values_clas[i].data)
        data_pos = pd.DataFrame(data={'values':values,'numbers':numbers}, index=keys).sort_values(by = "values", ascending=False)[0:5]
        data_neg = pd.DataFrame(data={'values':values,'numbers':numbers}, index=keys).sort_values(by = "values", ascending=True)[0:5]
        
        print(' ')
        print('What makes it slippery:')
        print(' ')
        print(data_pos.loc[:,'numbers'])
        
        print(' ')
        print('What makes it less slippery:')
        print(' ')
        print(data_neg.loc[:,'numbers'])

#-----------------------  Global    -----------------------#
# Print plot of global SHAP values for the test set

if cfg.shap == 'global':
    
    #Regression
    feat_df = X_test_reg.copy()
    
    # Rename columns for nicer plots
    feat_df = feat_df.rename(columns={'depos_2.0': 'Contam. Wet', 'accum_ds_24hour': 'Accum. Dry Snow 24h', 'depth'
                                      : 'Contam. Depth', 'hum_24hour':'Humidity 24h','dp_12hour':'Dew Point 12h',
                                      'rvr':'Horizontal Visibility','aerodrome':'Airport Runway','al_calc02_in':'Along Wind',
                                      'depos_1.0':'Contam. Damp','tmp_abs':'Absolute Air Temp.',
                                      'depos_47.0':'Contam. Dry Snow / Ice','rwy_24hour':'Runway Temp. 24h',
                                      'rwy_1hour':'Runway Temp. 1h','tmp_12hour':'Air Temp. 12h',
                                      'hum_1hour': 'Humidity 1h','tmp_1hour':'Air Temp. 1h',
                                      'qnh_12hour':'Air Pressure 12h','qnh_3hour':'Air Pressure 3h',
                                      'rwy_12hour':'Runway Temp. 12h','rwy_3hour':'Runway Temp. 3h',
                                      'hum_3hour':'Humidity 3h','accum_ds_6hour':'Accum. Dry Snow 6h',
                                      'accum_ra_24hour':'Accum. Rain 24h','sand':'Sand','hum':'Humidity',
                                      'depos_0.0':'Contam. Bare and Dry','qnh_24hour':'Air Pressure 24h',
                                      'cov':'Contam. Coverage','rwy_diff_6hour':'Runway Temp. Diff. 6h',
                                      'rwy_6hour':'Runway Temp. 6h','oeye_3hour_0.0':'Precipation 3h None',
                                      'ac_calc02_in':'Across Wind','rwy_diff_12hour':'Runway Temp. Diff. 12h',
                                      'qnh_diff_12hour':'Air Pressure Diff. 12h','rwy':'Runway Temp.', 
                                      'dp_6hour':'Dew Point 6h','pcpint_1hour':'Precipitation Intensity 1h',
                                      'across_wind_abs':'Abs Across Wind','qnh_diff_24hour':'Air Pressure Diff. 24h'})
    
    shap_values_1 = explainer_reg.shap_values(X_test_reg,check_additivity = False)
    shap.summary_plot(shap_values_1, feat_df)
    
    # CLassification
    feat_df = X_test_clas.copy()
    
    # Rename columns for nicer plots
    feat_df = feat_df.rename(columns={'depos_2.0': 'Contam. Wet', 'accum_ds_24hour': 'Accum. Dry Snow 24h', 'depth'
                                      : 'Contam. Depth', 'hum_24hour':'Humidity 24h','dp_12hour':'Dew Point 12h',
                                      'rvr':'Horizontal Visibility','aerodrome':'Airport Runway','al_calc02_in':'Along Wind',
                                      'depos_1.0':'Contam. Damp','tmp_abs':'Absolute Air Temp.',
                                      'depos_47.0':'Contam. Dry Snow / Ice','rwy_24hour':'Runway Temp. 24h',
                                      'rwy_1hour':'Runway Temp. 1h','tmp_12hour':'Air Temp. 12h',
                                      'hum_1hour': 'Humidity 1h','tmp_1hour':'Air Temp. 1h',
                                      'qnh_12hour':'Air Pressure 12h','qnh_3hour':'Air Pressure 3h',
                                      'rwy_12hour':'Runway Temp. 12h','rwy_3hour':'Runway Temp. 3h',
                                      'hum_3hour':'Humidity 3h','accum_ds_6hour':'Accum. Dry Snow 6h',
                                      'accum_ra_24hour':'Accum. Rain 24h','sand':'Sand','hum':'Humidity',
                                      'depos_0.0':'Contam. Bare and Dry','qnh_24hour':'Air Pressure 24h',
                                      'cov':'Contam. Coverage','rwy_diff_6hour':'Runway Temp. Diff. 6h',
                                      'rwy_6hour':'Runway Temp. 6h','oeye_3hour_0.0':'Precipation 3h None',
                                      'ac_calc02_in':'Across Wind','rwy_diff_12hour':'Runway Temp. Diff. 12h',
                                      'qnh_diff_12hour':'Air Pressure Diff. 12h','rwy':'Runway Temp.', 
                                      'dp_6hour':'Dew Point 6h','pcpint_1hour':'Precipitation Intensity 1h',
                                      'across_wind_abs':'Abs Across Wind','qnh_diff_24hour':'Air Pressure Diff. 24h'})
    
    shap_values = explainer_clas.shap_values(X_test_clas,check_additivity = False)
    shap.summary_plot(shap_values, feat_df)