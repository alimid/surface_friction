# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:45:05 2020

@author: Alise Danielle Midtfjord

This script works as a config file. 
"""

# path to csv file
path = 'df_jupyter.csv'

# Random state for splitting of data and classification
state = 333

# Name og column indicating wheter a landing is friction limited or not (1 = friction limited)
fricLim = 'filterRaw'

# Name of column with the friction coefficient
mu = 'mu_allRaw'


unusable_columns = ['filterRaw', 'airp','filterCal','t2g_allCal_tr', 'index', 'type', 'timediff',
                   'scen1','scen2','scen3','scen4','scen5','scen6','scen7','scen8',
                   'fricLim', 'fltId','ba', 'swen', 'age','cov','date','rwy2','tmp2'
  #                 ,'mu_allRaw'
                   ]

shap = 'local' # global or local

# Parameters for XGBoost classification done prior to calculating SHAP values
learning_rate_clas = 0.05
min_split_loss_clas = 0.26
reg_lambda_clas = 3.68
subsample_clas = 0.48
num_round_clas = 204

# Parameters for XGBoost regression done prior to calculating SHAP values
learning_rate_reg = 0.03
min_split_loss_reg = 0.02
reg_lambda_reg = 0.87
subsample_reg = 0.85
num_round_reg = 162
