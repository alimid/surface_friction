    # -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:33:41 2020

@author: Alise Danielle Midtfjord

This script works as a package used in , and contains functions used for 
machine learning / classification of the flight landings. The classifications are binary and predicts 
if a landing is friction limited or not. 

"""

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import configs as cfg
    
    
def draw_roc_curve(fpr,tpr,result=None):
    '''
    Draws and ROC curve from false positive rate and true positive rate
    
    :param Array fpr: False positive rate
    :param Array tpr: True positive rate
    :param int result: ROC AUC result
    '''
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw ,label='ROC curve (area = %0.2f)' % result)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
def classification(dtest,dtrain,y_train, y_test):
    '''
    Perfom an XGBoost classification according to chosen parameters
    
    :param Dataframe dtest: XGBoost dMatrix with test explanatory variables
    :param Dataframe dtrain: XGBoost dMatrix with train explanatory variables
    :param Array y_train: Array with train response
    :param Array y_test: DArray with test response
    :return: The boosting model
    :return: The prediction on the test data
    :return: The threshold for classification
    '''
    
    param = {'objective': 'binary:logistic'}
    param['eval_metric'] = 'auc'

    param['learning_rate'] = cfg.learning_rate_clas
    param['min_split_loss'] = cfg.min_split_loss_clas
    param['reg_lambda'] = cfg.reg_lambda_clas
    param['subsample'] = cfg.subsample_clas
    num_round = cfg.num_round_clas
    

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    
    bst = xgb.train(param, dtrain, num_round, evallist)
    
                   
    y_pred = bst.predict(dtest)
    y_proba = y_pred
    threshold = sum(y_train)/len(y_train)
    auc_score = roc_auc_score(y_test,y_pred)
    
    
    y_class = y_pred.copy()
    y_class[y_class<threshold]=0
    y_class[y_class!=0]=1
    y_test = y_test.reset_index(drop = True)
    
    print(' ')
    conf_matrix = confusion_matrix(y_test, y_class)
    print('     |              Predicted')
    print('-----|------------------------------------')
    print('     |             | Not FricLim | Friclim')
    print('     |------------------------------------')
    print('True | Not Friclim |' + str(conf_matrix[0][0]) + '         | ' + str(conf_matrix[0][1]))
    print('     |------------------------------------')
    print('     |     Friclim |' + str(conf_matrix[1][0]) + '           | ' + str(conf_matrix[1][1]))
    print('     |------------------------------------')
    print(' ')
    
    
    error =  y_test-y_class
    print('Accuracy:', (error.value_counts()[0] / len(error)).round(4))
    print('ROC AUC:', (auc_score).round(4))
    
    
    fpr, tpr, thresholds= roc_curve(y_test, y_proba)
    draw_roc_curve(fpr,tpr, result = auc_score)
    return(bst,y_pred, threshold)


def regression(dtest,dtrain,y_train, y_test):
    '''
    Perfom an XGBoost regression according to chosen parameters
    
    :param Dataframe dtest: XGBoost dMatrix with test explanatory variables
    :param Dataframe dtrain: XGBoost dMatrix with train explanatory variables
    :param Array y_train: Array with train response
    :param Array y_test: DArray with test response
    :return: The boosting model
    :return: The prediction on the test data
    '''
    
    param = {'objective': 'reg:squarederror'}
    param['eval_metric'] = 'rmse'
    param['learning_rate'] = cfg.learning_rate_reg
    param['min_split_loss'] = cfg.min_split_loss_reg
    param['reg_lambda'] = cfg.reg_lambda_reg
    param['subsample'] = cfg.subsample_reg
    

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    
    num_round = cfg.num_round_reg
    bst = xgb.train(param, dtrain, num_round, evallist)
    
                   
    y_pred = bst.predict(dtest)
    
    
    y_test_cat = pd.cut(x = y_test, bins = [-1, 0.05,0.075,0.1,0.15,0.2,100],right = True, labels = [0,1,2,3,4,5]) 
    y_pred_cat = pd.cut(x = pd.Series(y_pred), bins = [-1, 0.05,0.075,0.1,0.15,0.2,100],right = True, labels = [0,1,2,3,4,5]) 
    
    y_test_cat = y_test_cat.reset_index(drop = True).astype('int')
    y_pred_cat = y_pred_cat.reset_index(drop = True).astype('int')
    
    y_diff = y_test_cat-y_pred_cat
    y_diff_abs = abs(y_diff)
    
    print(' ')
    print('Test RMSE:',round(mean_squared_error(y_test, y_pred, squared= False),4))
    print('Test mean value:',round(y_test.mean(),4))
    print('Test RMSE / mean value:',round(mean_squared_error(y_test, y_pred, squared= False)/y_test.mean(),4))
    print(' ')
    print('Test class error:',round(y_diff_abs.mean(),4))
    
    print('0 : {:.0f} : {:.1f}%'.format(y_diff_abs.value_counts()[0],100*y_diff_abs.value_counts()[0]/len(y_test)))
    print('1 : {:.0f} : {:.1f}%'.format(y_diff_abs.value_counts()[1],100*y_diff_abs.value_counts()[1]/len(y_test)))
    print('2 : {:.0f} : {:.1f}%'.format(y_diff_abs.value_counts()[2],100*y_diff_abs.value_counts()[2]/len(y_test)))
    print('3 : {:.0f} : {:.1f}%'.format(y_diff_abs.value_counts()[3],100*y_diff_abs.value_counts()[3]/len(y_test)))
    print('4 : {:.0f} : {:.1f}%'.format(y_diff_abs.value_counts()[4],100*y_diff_abs.value_counts()[4]/len(y_test)))
    return(bst, y_pred)

