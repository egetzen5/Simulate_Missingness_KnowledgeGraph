import math
import sys
from sklearn.linear_model import LogisticRegression
import random
import os
import csv
import json
from collections import defaultdict
import gensim
import numpy as np
import pandas as pd
global metrics
import sklearn.metrics as metrics
import random 
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV
import sklearn.metrics as metrics
from sklearn import metrics
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
import os
import tensorflow as tf
import numpy as np
import scipy.spatial.distance as ds
from numpy import dot
from numpy.linalg import norm
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
import os
import pickle

import Run_KG_comparison_simulation
import Run_MAR_Simulations
import Run_MCAR_Simulations
import Run_MNAR_Simulations
import Run_stratified_experiments

data_path = '.../'
kg_path = '.../'
model_type = 'lasso' #or DL
test_type = 'incomplete' #do you want to remove data from test set or not
experiment = 'Knowledge graph'
demographic = 'gender' #for stratified experiments, which demographic variable do you want to see the impact of missing data?

if __name__ == "__main__":
    if experiment == 'Knowledge graph':
        Run_KG_comparison_simulation.run(data_path,kg_path,model_type,test_type)
    if experiment == 'MAR':
        Run_MAR_Simulations.run(data_path,kg_path,model_type,test_type)
    if experiment == 'MCAR':
        Run_MCAR_Simulations.run(data_path,model_type,test_type)
    if experiment == 'MNAR':
        Run_MNAR_Simulations.run(data_path,kg_path,model_type,test_type)
    if experiment == 'Stratified_Demographic':
        if demographic == 'gender':
            Run_stratified_experiments.run(data_path,kg_path,model_type,test_type,'gender')
        if demographic == 'race':  
            Run_stratified_experiments.run(data_path,kg_path,model_type,test_type,'race')
        if demographic == 'age': 
            Run_stratified_experiments.run(data_path,kg_path,model_type,test_type,'age2')
        if demographic == 'insurance':
            Run_stratified_experiments.run(data_path,kg_path,model_type,test_type,'insurance')










