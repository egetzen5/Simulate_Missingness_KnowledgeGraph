#Use with python/3.6.0 and tensorflow/1.10

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
#from glove import Corpus, Glove
#from bert_serving.client import BertClient
#bc = BertClient()
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


# fix random seed for reproducibility
#np.random.seed(7)

#from glove import Corpus, Glove
import osr
import pickle

lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)

ds = "mimic"
#data_path = "/Users/emily/Documents/SequentialPhenotypePredictor-master/Data/mimic_seq/"
data_path = '/home/egetzen/mimic_seq/'
window=10
size=100
decay=5
skipgram=1
norm=False

# list for files
train_files = []
valid_files = []
full_data_files = []


# In[2]:




for i in range(10):
    full_data_files.append(data_path + 'test_'+str(i))


    
    
events_files = []

for i in range(7):
    events_files.append(data_path + 'test_'+str(i))


    
count = -1
sss = []
sentences = []
for i in events_files:
    with open(i) as f:
        for s in f:
            count = count + 1
            sss.append(s)
            sentences.append(s.split("|")[2].split(" ") +

                             s.split("|")[3].replace("\n", "").split(" ")) 


# In[6]:


import itertools
flat_sent = list(itertools.chain(*sentences))
events2 = np.unique(flat_sent)
events2 = np.ndarray.tolist(events2)
len(events2)


# In[3]:


diag_totals = defaultdict(lambda: 0)
diag_totals2 = defaultdict(lambda: 0)
diag_joined = defaultdict(lambda: 0)
diag_joined2 = defaultdict(lambda:0)
test = defaultdict(lambda:0)
sentences = []
final_dx_train = []
disease_prev = []
list_duplicates = []
diag = []


# In[4]:


passed = ['d_250','d_585','d_428','d_403','d_272']


# In[153]:


pred = 'DL'
test_type = 'incomplete'
analysis = 'split'
condition = 'race'
file_path = '/home/egetzen/finalz_experiments1/'


#dice_mat = pd.read_csv("/users/emily/Documents/noisy_mat",header=None)
dice_mat = pd.read_csv("/home/egetzen/noisy_mat",header=None)

if pred == 'RNN':
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    
if analysis != 'split':
    if test_type == 'complete':
        if pred == 'lasso':
            path_auc = file_path + 'MARkg_lasso_aucc1'
            path_auc2 = file_path + 'MARkg_lasso_aucc2'
            path_ratios = file_path + 'MARkg_lasso_ratioc'
        if pred == 'DL':
            path_auc = file_path + 'MARkg_DL_aucc1'
            path_auc2 = file_path + 'MARkg_DL_aucc2'
            path_ratios = file_path + 'MARkg_DL_ratioc'
            
        if pred == 'RNN':
            path_auc = file_path + 'MARkg_RNN_aucc1'
            path_auc2 = file_path + 'MARkg_RNN_aucc2'
            path_ratios = file_path + 'MARkg_RNN_ratioc'
            
            
    if test_type == 'incomplete':
        if pred == 'lasso':
            path_auc = file_path + 'MARkg_lasso_auci1'
            path_auc2 = file_path + 'MARkg_lasso_auci2'
            path_ratios = file_path + 'MARkg_lasso_ratioi'
        if pred == 'DL':
            path_auc = file_path + 'MARkg_DL_auci1'
            path_auc2 = file_path + 'MARkg_DL_auci2'
            path_ratios = file_path + 'MARkg_DL_ratioi'
        if pred == 'RNN':
            path_auc = file_path + 'MARkg_RNN_auci1'
            path_auc2 = file_path + 'MARkg_RNN_auci2'
            path_ratios = file_path + 'MARkg_RNN_ratioi'
            

            
if analysis == 'split':
    if condition == 'insurance':

        if test_type == 'complete':

            if pred == 'lasso':
                path_auc = file_path + 'MARkgins_lasso_aucc1'
                path_auc2 = file_path + 'MARkgins_lasso_aucc2'
                path_ratios = file_path + 'MARkgins_lasso_ratioc'
                path_ratios2 = file_path + 'MARkgins_lasso_ratioc2'

            if pred == 'DL':
                path_auc = file_path + 'MARkgins_DL_aucc1'
                path_auc2 = file_path + 'MARkgins_DL_aucc2'
                path_ratios = file_path + 'MARkgins_DL_ratioc'
                file_path + 'MARkgins_DL_ratioc2'
                
            if pred == 'RNN':
                path_auc = file_path + 'MARkgins_RNN_aucc1'
                path_auc2 = file_path + 'MARkgins_RNN_aucc2'
                path_ratios = file_path + 'MARkgins_RNN_ratioc'

        if test_type == 'incomplete':

            if pred == 'lasso':
                path_auc = file_path + 'MARkgins_lasso_auci1'
                path_auc2 = file_path + 'MARkgins_lasso_auci2'
                path_ratios = file_path + 'MARkgins_lasso_ratioi'
                path_ratios2 = file_path + 'MARkgins_lasso_ratioi2'

            if pred == 'DL':
                path_auc = file_path + 'MARkgins_DL_auci1'
                path_auc2 = file_path + 'MARkgins_DL_auci2'
                path_ratios = file_path + 'MARkgins_DL_ratioi'
                path_ratios2 = file_path + 'MARkgins_DL_ratioi2'
                
            if pred == 'RNN':
                path_auc = file_path + 'MARkgins_RNN_auci1'
                path_auc2 = file_path + 'MARkgins_RNN_auci2'
                path_ratios = file_path + 'MARkgins_RNN_ratioi'





    if condition == 'gender':

        if test_type == 'complete':


            if pred == 'lasso':
                path_auc = file_path + 'MARkggender_lasso_aucc1'
                path_auc2 = file_path + 'MARkggender_lasso_aucc2'
                path_ratios = file_path + 'MARkggender_lasso_ratioc'

            if pred == 'DL':
                path_auc = '/home/egetzen/final_experiments/MARkggender_DL_aucc1'
                path_auc2 = '/home/egetzen/final_experiments/MARkggender_DL_aucc2'
                path_ratios = '/home/egetzen/final_experiments/MARkggender_DL_ratioc'
            
            if pred == 'RNN':
                path_auc = file_path + 'MARkggender_RNN_aucc1'
                path_auc2 = file_path + 'MARkggender_RNN_aucc2'
                path_ratios = file_path + 'MARkggender_RNN_ratioc'





        if test_type == 'incomplete':

  

            if pred == 'lasso':
                path_auc = file_path + 'MARkggender_lasso_auci1'
                path_auc2 = file_path + 'MARkggender_lasso_auci2'
                path_ratios = file_path + 'MARkggender_lasso_ratioi'
                path_ratios2 = file_path + 'MARkggender_lasso_ratioi2'

            if pred == 'DL':
                path_auc = file_path + 'MARkggender_DL_auci1'
                path_auc2 = file_path + 'MARkggender_DL_auci2'
                path_ratios = file_path + 'MARkggender_DL_ratioi'
                path_ratios2 = file_path + 'MARkggender_DL_ratioi2'
                
                
            if pred == 'RNN':
                path_auc = file_path + 'MARkggender_RNN_auci1'
                path_auc2 = file_path + 'MARkggender_RNN_auci2'
                path_ratios = file_path + 'MARkggender_RNN_ratioi'



    if condition == 'age2':
        if test_type == 'incomplete':

            if pred == 'lasso':
                path_auc = file_path + 'MARkgage_lasso_auci1'
                path_auc2 = file_path + 'MARkgage_lasso_auci2'
                path_ratios = file_path + 'MARkgage_lasso_ratioi'
                path_ratios2 = file_path + 'MARkgage_lasso_ratioi2'

            if pred == 'DL':
                path_auc = file_path + 'MARkgage_DL_auci1'
                path_auc2 = file_path + 'MARkgage_DL_auci2'
                path_ratios = file_path + 'MARkgage_DL_ratioi'
                path_ratios2 = file_path + 'MARkgage_DL_ratioi2'
                
            if pred == 'RNN':
                path_auc = file_path + 'MARkgage_RNN_auci1'
                path_auc2 = file_path + 'MARkgage_RNN_auci2'
                path_ratios = file_path + 'MARkgage_RNN_ratioi'


        if test_type == 'complete':

            if pred == 'lasso':
                path_auc = file_path + 'MARkgage_lasso_aucc1'
                path_auc2 = file_path + 'MARkgage_lasso_aucc2'
                path_ratios = file_path + 'MARkgage_lasso_ratioc'

            if pred == 'DL':
                path_auc = '/home/egetzen/final_experiments/MARkgage_DL_aucc1'
                path_auc2 = '/home/egetzen/final_experiments/MARkgage_DL_aucc2'
                path_ratios = '/home/egetzen/final_experiments/MARkgage_DL_ratioc'
                
            if pred == 'RNN':
                path_auc = file_path + 'MARkgage_RNN_aucc1'
                path_auc2 = file_path + 'MARkgage_RNN_aucc2'
                path_ratios = file_path + 'MARkgage_RNN_ratioc'
                
    if condition == 'race':
        if test_type == 'incomplete':

            if pred == 'lasso':
                path_auc = file_path + 'MARkgrace_lasso_auci1'
                path_auc2 = file_path + 'MARkgrace_lasso_auci2'
                path_ratios = file_path + 'MARkgrace_lasso_ratioi'
                path_ratios2 = file_path + 'MARkgrace_lasso_ratioi2'

            if pred == 'DL':
                path_auc = file_path + 'MARkgrace_DL_auci1'
                path_auc2 = file_path + 'MARkgrace_DL_auci2'
                path_ratios = file_path + 'MARkgrace_DL_ratioi'
                path_ratios2 = file_path + 'MARkgrace_DL_ratioi2'
                
            if pred == 'RNN':
                path_auc = file_path + 'MARkgrace_RNN_auci1'
                path_auc2 = file_path + 'MARkgrace_RNN_auci2'
                path_ratios = file_path + 'MARkgrace_RNN_ratioi'


        if test_type == 'complete':

            if pred == 'lasso':
                path_auc = file_path + 'MARkgrace_lasso_aucc1'
                path_auc2 = file_path + 'MARkgrace_lasso_aucc2'
                path_ratios = file_path + 'MARkgrace_lasso_ratioc'

            if pred == 'DL':
                path_auc = file_path + 'MARkgrace_DL_aucc1'
                path_auc2 = file_path + 'MARkgrace_DL_aucc2'
                path_ratios = '/home/egetzen/final_experiments/MARkgrace_DL_ratioc'
                
            if pred == 'RNN':
                path_auc = file_path + 'MARkgrace_RNN_aucc1'
                path_auc2 = file_path + 'MARkgrace_RNN_aucc2'
                path_ratios = file_path + 'MARkgrace_RNN_ratioc'



dat = list(range(17,100,1))




dataset_auc = []
dataset_auc2 = []
dataset_auc3 = []
dataset_auc4 = []
ratios_dis = []
ratios_dis2 = []

for q in range(len(passed)):  
    AUC = []
    AUC2 = []
    AUC3 = []
    AUC4 = []
    ratios_prop = []
    ratios_prop2 = []


    #prop = [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,0.55,0.6]
    #prop = [0,0.09,.18,.27,.36,0.45]
    #prop = [0,.06,.18,.48]
    #prop = [0,0.05,.16,.32,.48]
    #prop =  [0.05,.16,.32,.48]
    prop = [0,0,0,0,0]
    prob_marg = [.02,.095,.195,.33,.49]
    prob_nomarg = [.02,.02,.02,.02,.02]

    for h in range(len(prop)):
        p = prop[h]
        AUC_hold = []
        AUC_hold2 = []
        AUC_hold3 = []
        AUC_hold4 = []
        ratios_hold = []
        ratios_hold2 = []
        for w in range(100):
            valid_files = []
            
            p = prop[h]
            iters = w
            random.seed(range(200)[iters])
            
            
            train_files = random.sample(full_data_files,7)
            for i in range(len(full_data_files)):
                if full_data_files[i] not in train_files:
                    valid_files.append(full_data_files[i])
                    



            # In[5]:


            count = -1
            sss = []
            sentences = []
            for i in train_files:
                with open(i) as f:
                    for s in f:
                        count = count + 1
                        sss.append(s)
                        sentences.append(s.split("|")[2].split(" ") +

                                         s.split("|")[3].replace("\n", "").split(" ")) 


            # In[6]:


            import itertools
            flat_sent = list(itertools.chain(*sentences))
            events = np.unique(flat_sent)
            events = np.ndarray.tolist(events)
            len(events)

                

    ## Identify neonates to be excluded, extract sentences with all events in medical record, and target diagnoses existing in history


            def prep_data(data_files,iters):
                disease_prev = []
                omit1 = []
                sentences = []
                age_dyn = []
                gender = []
                count = -1
                omit2 = []
                Xins = []
                Xgen = []
                Xage = []
                Xwhite = []
                disease_prev = []
                omit1 = []
                count = -1
                omit2 = []
                omit3 = []
                omit4 = []
                X_marg = []


                for i in data_files:
                    with open(i) as f:
                        for s in f:
                            count = count+1
                            age = float(s.split("|")[1].split(" ")[3].replace(",",r""))
                            gen = int(s.split("|")[1].split(" ")[1].replace(",",r""))
                            insurance = s.split("|")[1].split(" ")[5].replace(",",r"")
                            if insurance != '"Self':
                                white = int(s.split("|")[1].split(" ")[7].replace(",",r""))
                                black = int(s.split("|")[1].split(" ")[13].replace(",",r""))
                            else:
                                white = int(s.split("|")[1].split(" ")[8].replace(",",r""))
                                black = int(s.split("|")[1].split(" ")[14].replace(",",r""))
                            if insurance == '"Medicare"' or insurance == '"Medicaid"' or insurance == '"Government"':
                                govt = 1
                            else:
                                govt = 0
                            if white == 0 and black == 0:
                                other = 1
                            else:
                                other = 0
                            if insurance == '"Self':
                                other2 = 1
                            else:
                                other2 = 0
                            if age <= 5:
                                age_dyn.append(0)
                            if age > 5 and age <= 55:
                                age_dyn.append(1)
                            if age > 55:
                                age_dyn.append(2)


                            if insurance == '"Medicare"' or insurance == '"Medicaid"' or insurance == '"Government"':
                                Xins.append(1)
                                if condition == 'insurance':
                                    X_marg.append(1)
                            else:
                                Xins.append(0)
                                if condition == 'insurance':
                                    X_marg.append(0)


                            gender.append(int(s.split("|")[1].split(" ")[1].replace(",",r"")))
                            if gen == 1:
                                Xgen.append(1)
                                if condition == 'gender':
                                    X_marg.append(1)
                            else:
                                Xgen.append(0)
                                if condition == 'gender':
                                    X_marg.append(0)

                            if age <= 55:
                                Xage.append(1)
                                if condition == 'age2':
                                    X_marg.append(1)
                            else:
                                Xage.append(0)
                                if condition == 'age2':
                                    X_marg.append(0)

                            Xage.append(((age-np.mean(dat))/(.5*np.std(dat)))-.9)

                            if white == 0:
                                Xwhite.append(0)
                                if condition == 'race':
                                    X_marg.append(1)
                            else:
                                Xwhite.append(1)
                                if condition == 'race':
                                    X_marg.append(0)

                            if data_files == train_files:
                                if age <= 5 or insurance == '"Self':
                                    omit1.append(1)
                                else:
                                    omit1.append(0)
                            if data_files == valid_files:
                                if analysis == 'split':
                                    if condition == 'gender':                                
                                        if age <= 5 or gen== 0:
                                            omit1.append(1)
                                        else:
                                            omit1.append(0)
                                        if age <= 5 or gen == 1:
                                            omit2.append(1)                                
                                        else:
                                            omit2.append(0)
                                    if condition == 'race':                                
                                        if age <= 5 or white== 1 or other == 1:
                                            omit1.append(1)
                                        else:
                                            omit1.append(0)
                                        if age <= 5 or black == 1 or other == 1:
                                            omit2.append(1)                                
                                        else:
                                            omit2.append(0)
                                    if condition == 'insurance':                                
                                        if age <= 5 or insurance == '"Self' or govt == 0:
                                            omit1.append(1)
                                        else:
                                            omit1.append(0)
                                        if age <= 5 or insurance == '"Self' or govt == 1:
                                            omit2.append(1)                                
                                        else:
                                            omit2.append(0)
                                    if condition == 'age2':                                
                                        if age <= 5 or age > 55:
                                            omit1.append(1)
                                        else:
                                            omit1.append(0)
                                        if age <= 5 or age <= 55:
                                            omit2.append(1)                                
                                        else:
                                            omit2.append(0)



                            prev_diags = [e for e in s.split("|")[2].split(" ") if e.startswith("d_")]
                            if passed[q] in prev_diags: # if disease exit in the previous diagnosis
                                disease_prev.append(1)
                            else:
                                disease_prev.append(0)
                            sentences.append(s.split("|")[2].split(" ") +
                                                 s.split("|")[3].replace("\n", "").split(" "))
                            B1 = -0.62
                            B2 = -0.62
                            B3 = -1.2
                            B4 = -0.62

                p_select = [1 for e in range(len(age_dyn))]





                ran = range(200)[iters]
                random.seed(ran)
                np.random.seed(ran)
                tf.random.set_seed(ran)


                remove_patient_data = [int(np.random.binomial(1,p_select[e],1)) for e in range(len(sentences))]



                return sentences, omit1, omit2, remove_patient_data, disease_prev,X_marg
            sentences, omit1, omit2,remove_patient_train,  disease_prev,X_marg = prep_data(train_files,iters)

            

            ## Within each sentence, split medical record into individual visits  
            def split_sentences(sentences):
                newsents = []
                for count in range(len(sentences)):
                    partials = []    
                    z = 0
                    for i in range(len(sentences[count])-1):        
                        i = i+1  
                        if (sentences[count])[i-1].startswith("d") or (i == len(sentences[count])-1):             
                            if not sentences[count][i].startswith("d") or (i == len(sentences[count])-1):
                                if (i == len(sentences[count])-1):
                                    part = sentences[count][z:i+1]
                                    partials.append(part)
                                else:
                                    part = sentences[count][z:i]
                                    partials.append(part)
                                    z = i

                        if (sentences[count])[i-1].startswith("s"):
                            if not sentences[count][i].startswith("d"): 
                                if not sentences[count][i].startswith("s"):
                                    if (i == len(sentences[count])-1):
                                        part = sentences[count][z:i+1]
                                        partials.append(part)
                                    else:
                                        part = sentences[count][z:i]
                                        partials.append(part)
                                        z = i

                        if (sentences[count])[i-1].startswith("c"):
                            if not sentences[count][i].startswith("d"):
                                if not sentences[count][i].startswith("s"): 
                                    if not sentences[count][i].startswith("c"):
                                        if (i == len(sentences[count])-1):
                                            part = sentences[count][z:i+1]
                                            partials.append(part)
                                        else:
                                            part = sentences[count][z:i]
                                            partials.append(part)
                                            z = i

                    newsents.append(partials)
                return newsents
            newsents = split_sentences(sentences)

            ## Within each sentence, split medical record into individual visits  
            def split_sentences2(sentences):
                newsents = []
                for count in range(len(sentences)):
                    partials = []    
                    z = 0
                    for i in range(len(sentences[count])-1):        
                        i = i+1  
                        if (sentences[count])[i-1].startswith("d_"):             
                            if not sentences[count][i].startswith("d_") or (i == len(sentences[count])-1):
                                part = sentences[count][z:i]
                                partials.append(part)
                                z = i

                        if (sentences[count])[i-1].startswith("s"):
                            if not sentences[count][i].startswith("d_"): 
                                if not sentences[count][i].startswith("s") or (i == len(sentences[count])-1):
                                    part = sentences[count][z:i]
                                    partials.append(part)
                                    z = i

                        if (sentences[count])[i-1].startswith("c"):
                            if not sentences[count][i].startswith("d_"):
                                if not sentences[count][i].startswith("s"): 
                                    if not sentences[count][i].startswith("c") or (i == len(sentences[count])-1):
                                        part = sentences[count][z:i]
                                        partials.append(part)
                                        z = i

                    newsents.append(partials)
                return newsents
            newsents2 = split_sentences2(sentences)

            #target_loc = []
            def adjust_all(newsents):
                # Identify the visit in the history where the target diagnosis occurs, if it does                                    
                target_loc = []
                ct1 = 0
                ct2 = 0
                for i in range(len(sentences)):
                    ct = 0
                    if passed[q] in list(np.concatenate(newsents[i])):
                        for j in range(len(newsents[i])):
                            if ct == 0 and passed[q] in (newsents[i])[j]:
                                target_loc.append(j)
                                ct = ct + 1
                                ct1 = ct1+1
                    else:
                        target_loc.append(100)
                        ct2 = ct2+1
                return target_loc
            target_loc = adjust_all(newsents)

            #data refers to train or test 
            def adjust_exclusions(sentences, newsents,omit):                    
                # create a vector indicating which patients are positive for target diagnosis

                disease_data = []
                for count in range(len(sentences)):  
                    if passed[q] in sentences[count]:
                        disease_data.append(1)
                    else:
                        disease_data.append(0)


                #exclude patients with target diagnosis present in first visit
                exc = []                
                for ii in range(len(sentences)): 
                    if target_loc[ii]==0:
                        exc.append(1)
                    else:
                        exc.append(0)

                #Update vector containing patients to exclude
                for count in range(len(sentences)):
                    if exc[count]==1:
                        omit[count] = 1

                return disease_data, omit

            disease_train, omit1 = adjust_exclusions(sentences,newsents,omit1)


        #Convert events in history to patient vector


            def patient_vec(data_files, newsents, remove_patient_data, disease_prev, target,omit,p,iters,X_marg):
                
                ran = range(200)[iters]
                random.seed(ran)
                np.random.seed(ran)
                tf.random.set_seed(ran)
                
                new_sentences = []
                count = -1
                patient_seq_all_disease = []
                exclude = []
                global model
                global ratio
                before = []
                after = []
                for i in data_files: 
                    with open(i) as f:
                        for line in f:
                            count = count + 1
                            patient_seq = []
                            #feed_events = line.split("|")[2].split(" ")
                            if omit[count]==1 or newsents[count]==[] or newsents[count][0:-1] == []:
                                exclude.append(1)
                            else:
                                if disease_prev[count] == 1:  
                                    visits = newsents[count][:target[count]]

                                    if remove_patient_data[count]==1:
                                        if X_marg[count] == 1:
                                            p = np.random.normal(prob_marg[h],.05,1)
                                            if p < 0:
                                                p = 0
                                            if p > 1:
                                                p = 1
                                        else:
                                            p = np.random.normal(prob_nomarg[h],.05,1)
                                            if p < 0:
                                                p = 0
                                            if p > 1:
                                                p = 1
                                        
                                        before.append(len(list(np.concatenate(visits))))
                                        new_visits_all = []
                                        test = np.random.binomial(1,p,len(visits))
                                        remm = [b for b in range(len(test)) if test[b] == 1]
                                        newv = [s for s in visits if visits.index(s) not in remm]
                                        visits = newv
                                        if visits == []:
                                            exclude.append(1)
                                        else:
                                            for i in range(len(visits)):
                                                new_visits = visits[i]
                                                removals = []
                                                visits_unique = np.unique(visits[i])
                                                visits_unique = np.ndarray.tolist(visits_unique)
                                                new_visits_unique = np.unique(visits[i])
                                                new_visits_unique = np.ndarray.tolist(new_visits_unique)
                                                if 1-p != 1:
                                                    while len(new_visits_unique)/len(visits_unique)>= 1-p:
                                                        if removals == []:
                                                            rem = (random.sample(visits_unique,1))[0]
                                                            if rem in events2:
                                                                removals.append(rem)
                                                                probabilities = []
                                                                recent_removals = []
                                                                for w in range(len(visits_unique)):
                                                                    if visits_unique[w] in events2:
                                                                        if np.random.binomial(1,dice_mat.iloc[events2.index(rem),events2.index(visits_unique[w])],1)[0] == 1 and visits_unique[w] not in removals:
                                                                            removals.append(visits_unique[w])
                                                                            probabilities.append(dice_mat.iloc[events2.index(rem),events2.index(visits_unique[w])])
                                                                            recent_removals.append(visits_unique[w])
                                                                new_visits = [s for s in visits[i] if s not in removals]
                                                                if new_visits == []:
                                                                    #exclude.append(1)
                                                                    break
                                                                else:
                                                                    new_visits_unique = np.unique(new_visits)
                                                                    new_visits_unique = np.ndarray.tolist(new_visits_unique) 
                                                                    if recent_removals == []:
                                                                        rem2 = (random.sample(new_visits_unique,1))[0]
                                                                    else:
                                                                        rem2 = np.unique(recent_removals)[probabilities.index(max(probabilities))]
                                                            else:
                                                                removals.append(rem)
                                                                new_visits = [s for s in visits[i] if s not in removals]
                                                                if new_visits == []:
                                                                    #exclude.append(1)
                                                                    break
                                                                else:
                                                                    new_visits_unique = np.unique(new_visits)
                                                                    new_visits_unique = np.ndarray.tolist(new_visits_unique) 
                                                                    rem2 = (random.sample(new_visits_unique,1))[0]

                                                        else:
                                                            rem = rem2
                                                            if rem in events2:
                                                                removals.append(rem)
                                                                recent_removals = []
                                                                probabilities = []
                                                                for w in range(len(new_visits_unique)):
                                                                    if new_visits_unique[w] in events2:
                                                                        if np.random.binomial(1,dice_mat.iloc[events2.index(rem),events2.index(new_visits_unique[w])],1)[0] == 1 and new_visits_unique[w] not in removals:
                                                                            removals.append(new_visits_unique[w]) 
                                                                            probabilities.append(dice_mat.iloc[events2.index(rem),events2.index(new_visits_unique[w])])
                                                                            recent_removals.append(new_visits_unique[w])
                                                                new_visits = [s for s in visits[i] if s not in removals]
                                                                if new_visits == []:
                                                                    #exclude.append(1)
                                                                    break
                                                                else:
                                                                    new_visits_unique = np.unique(new_visits)
                                                                    new_visits_unique = np.ndarray.tolist(new_visits_unique) 
                                                                    if recent_removals == []:
                                                                        rem2 = (random.sample(new_visits_unique,1))[0]
                                                                    else:
                                                                        rem2 = np.unique(recent_removals)[probabilities.index(max(probabilities))]
                                                            else:
                                                                removals.append(rem)
                                                                new_visits = [s for s in visits[i] if s not in removals]
                                                                if new_visits == []:
                                                                    #exclude.append(1)
                                                                    break
                                                                else:
                                                                    new_visits_unique = np.unique(new_visits)
                                                                    new_visits_unique = np.ndarray.tolist(new_visits_unique) 
                                                                    rem2 = (random.sample(new_visits_unique,1))[0]
                                                new_visits_all.append(new_visits)
                                            visits = new_visits_all
                                            if new_visits_all == []:
                                                new_visits_all = visits
                                                exclude.append(1)
                                            else: 
                                                exclude.append(0)
                                                feed_events = list(np.concatenate(visits))
                                                new_sentences.append(feed_events)
                                                after.append(len(feed_events))


                                    else:
                                        if visits == []:
                                            exclude.append(1)
                                        else:
                                            exclude.append(0)
                                            feed_events = list(np.concatenate(visits))  
                                            new_sentences.append(feed_events)

                                else:
                                    if X_marg[count] == 1:
                                        p = np.random.normal(prob_marg[h],.05,1)
                                        if p < 0:
                                            p = 0
                                        if p > 1:
                                            p = 1
                                    else:
                                        p = np.random.normal(prob_nomarg[h],.05,1)
                                        if p < 0:
                                            p = 0
                                        if p > 1:
                                            p = 1
                                    visits = newsents[count][0:-1]  

                                    if remove_patient_data[count]==1:
                                        before.append(len(list(np.concatenate(visits))))
                                        new_visits_all = []
                                        test = np.random.binomial(1,p,len(visits))
                                        remm = [b for b in range(len(test)) if test[b] == 1]
                                        newv = [s for s in visits if visits.index(s) not in remm]
                                        visits = newv
                                        if visits == []:
                                            exclude.append(1)
                                        else:
                                            for i in range(len(visits)):
                                                new_visits = visits[i]
                                                removals = []
                                                visits_unique = np.unique(visits[i])
                                                visits_unique = np.ndarray.tolist(visits_unique)
                                                new_visits_unique = np.unique(visits[i])
                                                new_visits_unique = np.ndarray.tolist(new_visits_unique)
                                                if 1-p != 1:
                                                    while len(new_visits_unique)/len(visits_unique)>= 1-p:
                                                        if removals == []:
                                                            rem = (random.sample(visits_unique,1))[0]
                                                            if rem in events2:
                                                                removals.append(rem)
                                                                probabilities = []
                                                                recent_removals = []
                                                                for w in range(len(visits_unique)):
                                                                    if visits_unique[w] in events2:
                                                                        if np.random.binomial(1,dice_mat.iloc[events2.index(rem),events2.index(visits_unique[w])],1)[0] == 1 and visits_unique[w] not in removals:
                                                                            removals.append(visits_unique[w])
                                                                            probabilities.append(dice_mat.iloc[events2.index(rem),events2.index(visits_unique[w])])
                                                                            recent_removals.append(visits_unique[w])
                                                                new_visits = [s for s in visits[i] if s not in removals]
                                                                if new_visits == []:
                                                                    #exclude.append(1)
                                                                    break
                                                                else:
                                                                    new_visits_unique = np.unique(new_visits)
                                                                    new_visits_unique = np.ndarray.tolist(new_visits_unique) 
                                                                    if recent_removals == []:
                                                                        rem2 = (random.sample(new_visits_unique,1))[0]
                                                                    else:
                                                                        rem2 = np.unique(recent_removals)[probabilities.index(max(probabilities))]
                                                            else:
                                                                removals.append(rem)
                                                                new_visits = [s for s in visits[i] if s not in removals]
                                                                if new_visits == []:
                                                                    #exclude.append(1)
                                                                    break
                                                                else:
                                                                    new_visits_unique = np.unique(new_visits)
                                                                    new_visits_unique = np.ndarray.tolist(new_visits_unique) 
                                                                    rem2 = (random.sample(new_visits_unique,1))[0]

                                                        else:
                                                            rem = rem2
                                                            if rem in events2:
                                                                removals.append(rem)
                                                                recent_removals = []
                                                                probabilities = []
                                                                for w in range(len(new_visits_unique)):
                                                                    if new_visits_unique[w] in events2:
                                                                        if np.random.binomial(1,dice_mat.iloc[events2.index(rem),events2.index(new_visits_unique[w])],1)[0] == 1 and new_visits_unique[w] not in removals:
                                                                            removals.append(new_visits_unique[w]) 
                                                                            probabilities.append(dice_mat.iloc[events2.index(rem),events2.index(new_visits_unique[w])])
                                                                            recent_removals.append(new_visits_unique[w])
                                                                new_visits = [s for s in visits[i] if s not in removals]
                                                                if new_visits == []:
                                                                    #exclude.append(1)
                                                                    break
                                                                else:
                                                                    new_visits_unique = np.unique(new_visits)
                                                                    new_visits_unique = np.ndarray.tolist(new_visits_unique) 
                                                                    if recent_removals == []:
                                                                        rem2 = (random.sample(new_visits_unique,1))[0]
                                                                    else:
                                                                        rem2 = np.unique(recent_removals)[probabilities.index(max(probabilities))]
                                                            else:
                                                                removals.append(rem)
                                                                new_visits = [s for s in visits[i] if s not in removals]
                                                                if new_visits == []:
                                                                    #exclude.append(1)
                                                                    break
                                                                else:
                                                                    new_visits_unique = np.unique(new_visits)
                                                                    new_visits_unique = np.ndarray.tolist(new_visits_unique) 
                                                                    rem2 = (random.sample(new_visits_unique,1))[0]


                                                new_visits_all.append(new_visits)
                                            visits = new_visits_all
                                            if new_visits_all == []:
                                                new_visits_all = visits
                                                exclude.append(1) 
                                            else:
                                                exclude.append(0)
                                                feed_events = list(np.concatenate(visits))
                                                after.append(len(feed_events))
                                                new_sentences.append(feed_events)

                                    else:
                                        feed_events = list(np.concatenate(visits))  
                                        if feed_events==[]:
                                            exclude.append(1)
                                        else:
                                            exclude.append(0)
                                        feed_events = list(np.concatenate(visits)) 
                                        new_sentences.append(feed_events)




                ratio = 1-sum(after)/sum(before)
                

                return new_sentences, exclude, ratio
            
            
            


            #Update labels for presence of target diagnosis
            def update_data(exclude,disease_data,data_disease):                
                disease_final = []
                for i in range(0,len(exclude)):
                    if exclude[i] == 0:
                        disease_final.append(disease_data[i])

                data_disease[passed[q]] = disease_final

                X_data_disease=data_disease.iloc[:,0:100]
                Y_data_disease =data_disease.iloc[:,100]


                return X_data_disease, Y_data_disease

            def update_data_RNN(exclude,disease_data):
                datay = []
                for i in range(0,len(exclude)):
                    if exclude[i] == 0:
                        datay.append(disease_data[i])
                y_data = np.array(datay)
                


                
                return y_data

  


            new_sentences, exclude, ratio = patient_vec(train_files, newsents, remove_patient_train, disease_prev,target_loc,omit1,p,iters,X_marg)

            import os
            os.environ['PYTHONHASHSEED'] = '0'
            os.environ['CUDA_VISIBLE_DEVICES']='-1'
            os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'

            ran = range(200)[iters]
            random.seed(ran)
            np.random.seed(ran)
            tf.random.set_seed(ran)

            #print('fitting word2vec model')
            model = gensim.models.Word2Vec(new_sentences, sg=skipgram, window=window,
                                                 iter=5, size=size, min_count=1, workers=1)
            
            if pred == 'lasso' or pred == 'DL' or pred == 'PDPS':

                patient_seq_all_disease = []
                ct = -1
                for i in range(len(exclude)):
                    if exclude[i] == 0:
                        ct = ct + 1
                        te = len(new_sentences[ct])
                        weighted_events = [(e,  math.exp(decay*(j-te+1)/te)) for j, e in enumerate(new_sentences[ct]) if e in model.wv.vocab] 
                        if weighted_events == []:
                            exclude[i] = 1
                        else:
                            sum_weights = sum(weight for event,weight in weighted_events)
                            patient_seq = []
                            for a in weighted_events:
                                event, weight = a
                                patient_seq.append(weight*model.wv.word_vec(event,use_norm=norm))
                            patient_seq_all_disease.append(sum(patient_seq)/sum_weights)
                            
                data_disease=pd.DataFrame(patient_seq_all_disease)

                X_train_disease, Y_train_disease = update_data(exclude,disease_train,data_disease)


            if pred == 'RNN': 
                import itertools
                flat_sent2 = list(itertools.chain(*new_sentences))
                events2 = np.unique(flat_sent2)
                events2 = np.ndarray.tolist(events2)

                vecs = []
                for i in range(len(events2)):
                    vecs.append(model.wv.word_vec(events2[i],use_norm=norm))
                embeddings = np.asmatrix(vecs)


                X_train, exclude, ratio = patient_vec(train_files, newsents, remove_patient_train, disease_prev,target_loc,omit1,p,iters,X_marg)
                
                dict_events = {k: v for v, k in enumerate(events2)}
                X_train_ind = []
                for i in range(len(X_train)):
                    X_train_ind.append([dict_events[X_train[i][j]] for j in range(len(X_train[i])) if X_train[i][j] in events2])

                X_train = X_train_ind
 

                y_train = update_data_RNN(exclude, disease_train)


            if condition == 'insurance' or condition == 'gender' or condition == 'age2' or condition == 'race':


                sentences, omit1, omit2, remove_patient_test, disease_prev, X_marg2 = prep_data(valid_files,iters)
                
                

                newsents = split_sentences(sentences)

                newsents2 = split_sentences2(sentences)

                target_loc = adjust_all(newsents2)

                disease_test, omit1 = adjust_exclusions(sentences, newsents,omit1)

                if analysis == 'split':

                    disease_test2, omit2 = adjust_exclusions(sentences, newsents,omit2)

                if test_type == 'complete':
                    p = 0

                if pred == 'lasso' or pred == 'DL' or pred == 'PDPS':

                    new_sentences, exclude, ratio1 = patient_vec(valid_files, newsents,remove_patient_test,disease_prev, target_loc,omit1,p,iters,X_marg2)

                    if analysis == 'split':

                        new_sentences2, exclude2, ratio2 = patient_vec(valid_files, newsents,remove_patient_test,disease_prev, target_loc,omit2,p,iters,X_marg2)
                        
                    patient_seq_all_disease_test = []
                    ct = -1
                    for i in range(len(exclude)):
                        if exclude[i] == 0:
                            ct = ct + 1
                            te = len(new_sentences[ct])
                            weighted_events = [(e,  math.exp(decay*(j-te+1)/te)) for j, e in enumerate(new_sentences[ct]) if e in model.wv.vocab] 
                            if weighted_events == []:
                                exclude[i] = 1
                            else:
                                sum_weights = sum(weight for event,weight in weighted_events)
                                patient_seq = []
                                for a in weighted_events:
                                    event, weight = a
                                    patient_seq.append(weight*model.wv.word_vec(event,use_norm=norm))
                                patient_seq_all_disease_test.append(sum(patient_seq)/sum_weights)


                    if analysis == 'split':
                        patient_seq_all_disease_test2 = []
                        ct = -1
                        for i in range(len(exclude2)):
                            if exclude2[i] == 0:
                                ct = ct + 1
                                te = len(new_sentences2[ct])
                                weighted_events = [(e,  math.exp(decay*(j-te+1)/te)) for j, e in enumerate(new_sentences2[ct]) if e in model.wv.vocab] 
                                if weighted_events == []:
                                    exclude2[i] = 1
                                else:
                                    sum_weights = sum(weight for event,weight in weighted_events)
                                    patient_seq = []
                                    for a in weighted_events:
                                        event, weight = a
                                        patient_seq.append(weight*model.wv.word_vec(event,use_norm=norm))
                                    patient_seq_all_disease_test2.append(sum(patient_seq)/sum_weights)



                    data_disease_test = pd.DataFrame(patient_seq_all_disease_test)

                    if analysis == 'split':

                        data_disease_test2 = pd.DataFrame(patient_seq_all_disease_test2)

                    X_disease_test, Y_disease_test = update_data(exclude,disease_test,data_disease_test)

                    if analysis == 'split':

                        X_disease_test2, Y_disease_test2 = update_data(exclude2,disease_test2,data_disease_test2)


                if pred == 'RNN':



                    X_test, exclude, ratios2 = patient_vec(valid_files, newsents,remove_patient_test,disease_prev, target_loc,omit1,p,iters)
                    
          
                    X_test_ind = []
                    for i in range(len(X_test)):
                        X_test_ind.append([dict_events[X_test[i][j]] for j in range(len(X_test[i])) if X_test[i][j] in events2])

                        
                    X_test = X_test_ind
                    
                    if analysis == 'split':

                        X_test2, exclude2, ratios3 = patient_vec(valid_files, newsents, remove_patient_test,disease_prev, target_loc,omit2,p,iters)
                        
                        X_test_ind2 = []
                        for i in range(len(X_test2)):
                            X_test_ind2.append([dict_events[X_test2[i][j]] for j in range(len(X_test2[i])) if X_test2[i][j] in events2])

                        X_test2 = X_test_ind2

                    y_test= update_data_RNN(exclude,disease_test)

                    if analysis == 'split':

                        y_test2 = update_data_RNN(exclude2,disease_test2)


                if pred == 'lasso' or pred == 'DL' or pred == 'PDPS':


                    if pred == 'lasso':

                        if analysis == 'split':

                            if len(np.unique(Y_disease_test)) == 2 and len(np.unique(Y_disease_test2)) == 2:

                                #ratios_hold.append(ratio)
                                #print(ratio)
                                
                                ran = range(200)[iters]
                                random.seed(ran)
                                np.random.seed(ran)
                                tf.random.set_seed(ran)

                                searchCV_lasso = LogisticRegressionCV(Cs=list(np.power(10.0, np.arange(-10, 10))),penalty='l1',
                                cv=5,random_state=range(200)[iters],fit_intercept=True,solver='liblinear',max_iter = 10000,scoring="f1")
                                lasso_fit_disease = searchCV_lasso.fit(X_train_disease,Y_train_disease)
                                lasso_prob_disease = lasso_fit_disease.predict_proba(X_train_disease)[:,1]

                                lasso_prob_disease = lasso_fit_disease.predict_proba(X_disease_test)[:,1]
                                auc_disease_lasso = metrics.roc_auc_score(Y_disease_test, lasso_prob_disease)
                                AUC_hold.append(auc_disease_lasso)

                                if analysis == 'split':
                                    
                                    print(ratio1)
                                    print(ratio2)
                                    ratios_hold.append(ratio1)
                                    ratios_hold2.append(ratio2)

                                    lasso_prob_disease2 = lasso_fit_disease.predict_proba(X_disease_test2)[:,1]
                                    auc_disease_lasso2 = metrics.roc_auc_score(Y_disease_test2, lasso_prob_disease2)
                                    AUC_hold2.append(auc_disease_lasso2)
                                    ratios_hold2.append(ratio)





                        else:
                            
                            ran = range(200)[iters]
                            random.seed(ran)
                            np.random.seed(ran)
                            tf.random.set_seed(ran)

                            ratios_hold.append(ratio)
                            print(ratio)
                            searchCV_lasso = LogisticRegressionCV(Cs=list(np.power(10.0, np.arange(-10, 10))),penalty='l1',
                            cv=5,random_state = range(200)[iters],fit_intercept=True,solver='liblinear',max_iter = 10000,scoring="f1")
                            lasso_fit_disease = searchCV_lasso.fit(X_train_disease,Y_train_disease)
                            print(lasso_fit_disease.C_)
                            lasso_prob_disease = lasso_fit_disease.predict_proba(X_train_disease)[:,1]

                            lasso_prob_disease = lasso_fit_disease.predict_proba(X_disease_test)[:,1]
                            auc_disease_lasso = metrics.roc_auc_score(Y_disease_test, lasso_prob_disease)
                            AUC_hold.append(auc_disease_lasso)
                            print('AUC disease Dys:,',auc_disease_lasso)







                    if pred == 'DL':
                        
                        ran = range(200)[iters]
                        random.seed(ran)
                        np.random.seed(ran)
                        tf.random.set_seed(ran)

                        #from keras import backend as k
                        #config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        #allow_soft_placement=True, device_count = {'CPU': 1})
                        #sess = tf.Session(graph=tf.get_default_graph(),config=config)
                        #k.set_session(sess)

                        if analysis == 'split':


                            if len(np.unique(Y_disease_test)) == 2 and len(np.unique(Y_disease_test2)) == 2:
                                #ratios_hold.append(ratio)
                                print(ratio1)
                                print(ratio2)
                                ratios_hold.append(ratio1)
                                ratios_hold2.append(ratio2)
                                print("Reading files..."),
                                tr_dataframe = data_disease
                                te_dataframe = data_disease_test

                                if analysis == 'split':

                                    te_dataframe2 = data_disease_test2


                                print("Fineshed\n")

                                print("Compressing data..."),
                                tr_X = tr_dataframe.iloc[:, :-1].values
                                tr_Y = tr_dataframe.iloc[:, -1].values
                                te_X = te_dataframe.iloc[:, :-1].values
                                te_Y = te_dataframe.iloc[:, -1].values

                                if analysis == 'split':

                                    te_X2 = te_dataframe2.iloc[:, :-1].values
                                    te_Y2 = te_dataframe2.iloc[:, -1].values



                                # model_DL = KerasRegressor(build_fn = baseline_model(units = Grid.best_params_['units'], act = Grid.best_params_['act']), nb_epoch=2000, batch_size=300,verbose = 0)

                                model_DL = Sequential()
                                model_DL.add(Dense(55,activation = 'relu'))
                                model_DL.add(Dense(20,activation='relu'))
                                model_DL.add(Dense(1,activation = 'sigmoid'))
                                model_DL.compile(loss='mean_squared_error', optimizer='adam')



                                # In[ ]:




                                # In[32]:


                                model_DL.fit(tr_X,tr_Y,batch_size = 100, epochs=70, verbose = 0)
                                tr_predict_Y = model_DL.predict(tr_X)
                                te_predict_Y = model_DL.predict(te_X)    






                                # ROC AUC
                                auc = metrics.roc_auc_score(te_Y, te_predict_Y)
                                print('ROC AUC: %f' % auc)

                                AUC_hold.append(auc)

                                if analysis == 'split':

                                    te_predict_Y2 = model_DL.predict(te_X2) 

                                     # ROC AUC
                                    auc2 = metrics.roc_auc_score(te_Y2, te_predict_Y2)
                                    print('ROC AUC2: %f' % auc2)
                                    AUC_hold2.append(auc2)
                            else:
                                ratio = 'NA'

                        else:

                            ratios_hold.append(ratio)
                            print(ratio)
                            print("Reading files..."),
                            tr_dataframe = data_disease
                            te_dataframe = data_disease_test

                            if analysis == 'split':

                                te_dataframe2 = data_disease_test2


                            print("Fineshed\n")

                            print("Compressing data..."),
                            tr_X = tr_dataframe.iloc[:, :-1].values
                            tr_Y = tr_dataframe.iloc[:, -1].values
                            te_X = te_dataframe.iloc[:, :-1].values
                            te_Y = te_dataframe.iloc[:, -1].values

                            if analysis == 'split':

                                te_X2 = te_dataframe2.iloc[:, :-1].values
                                te_Y2 = te_dataframe2.iloc[:, -1].values



                            # model_DL = KerasRegressor(build_fn = baseline_model(units = Grid.best_params_['units'], act = Grid.best_params_['act']), nb_epoch=2000, batch_size=300,verbose = 0)

                            model_DL = Sequential()
                            model_DL.add(Dense(55,activation = 'relu'))
                            model_DL.add(Dense(20,activation='relu'))
                            model_DL.add(Dense(1,activation = 'sigmoid'))
                            model_DL.compile(loss='mean_squared_error', optimizer='adam')



                            # In[ ]:




                            # In[32]:


                            model_DL.fit(tr_X,tr_Y,batch_size = 100, epochs=70, verbose = 0)
                            tr_predict_Y = model_DL.predict(tr_X)
                            te_predict_Y = model_DL.predict(te_X)    






                            # ROC AUC
                            auc = metrics.roc_auc_score(te_Y, te_predict_Y)
                            print('ROC AUC: %f' % auc)

                            AUC_hold.append(auc)





                    if pred == 'PDPS':

                        if len(np.unique(Y_disease_test)) == 2 and len(np.unique(Y_disease_test2)) == 2:

                            if analysis == 'split':

                                ratios_prop.append(ratio)
                                print(ratio)
                                cos_sim_test = []
                                f1_calc = []
                                for count in range(len(patient_seq_all_disease_test)): 
                                    a = patient_seq_all_disease_test[count]
                                    b = model.wv.word_vec(passed[q],use_norm=norm)
                                    norma = np.sqrt(a.dot(a))
                                    normb = np.sqrt(b.dot(b))
                                    cos_sim_test.append(dot(a, b)/(norma*normb))

                                aucc=(metrics.roc_auc_score(Y_disease_test, cos_sim_test))
                                AUC.append(metrics.roc_auc_score(Y_disease_test, cos_sim_test))
                                print('AUC disease Dys:,',aucc)

                                if analysis == 'split' and test_type == 'incomplete':

                                    cos_sim_test2 = []
                                    f1_calc = []
                                    for count in range(len(patient_seq_all_disease_test2)): 
                                        a = patient_seq_all_disease_test2[count]
                                        b = model.wv.word_vec(passed[q],use_norm=norm)
                                        norma = np.sqrt(a.dot(a))
                                        normb = np.sqrt(b.dot(b))
                                        cos_sim_test2.append(dot(a, b)/(norma*normb))

                                    aucc2=(metrics.roc_auc_score(Y_disease_test2, cos_sim_test2))
                                    AUC2.append(metrics.roc_auc_score(Y_disease_test2, cos_sim_test2))
                                    print('AUC disease Dys:,',aucc2)

                            else:
                                ratio = 'NA'

                        else:

                            ratios_prop.append(ratio)
                            print(ratio)
                            cos_sim_test = []
                            f1_calc = []
                            for count in range(len(patient_seq_all_disease_test)): 
                                a = patient_seq_all_disease_test[count]
                                b = model.wv.word_vec(passed[q],use_norm=norm)
                                norma = np.sqrt(a.dot(a))
                                normb = np.sqrt(b.dot(b))
                                cos_sim_test.append(dot(a, b)/(norma*normb))

                            aucc=(metrics.roc_auc_score(Y_disease_test, cos_sim_test))
                            AUC.append(metrics.roc_auc_score(Y_disease_test, cos_sim_test))
                            print('AUC disease Dys:,',aucc)



                if pred == 'RNN':
                    
                    ran = range(150)[iters]
                    random.seed(ran)
                    np.random.seed(ran)
                    tf.random.set_seed(ran)
                    
                    
                    if analysis=='split':

                        if len(np.unique(y_test)) == 2 and len(np.unique(y_test2)) == 2:
                            y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
                            y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
                            y_test2 = np.asarray(y_test2).astype('float32').reshape((-1,1))
                            
                            ratios_prop.append(ratio)
                            print(ratio)


                            # truncate and pad input sequences
                            max_review_length = 3377
                            X_train1 = sequence.pad_sequences(X_train, maxlen=max_review_length)
                            X_test1 = sequence.pad_sequences(X_test, maxlen=max_review_length)

                            if analysis == 'split':

                                X_test2 = sequence.pad_sequences(X_test2, maxlen=max_review_length)

                            import sklearn.metrics as metrics
                            from sklearn.metrics import f1_score


                            # create the model
                            embedding_vecor_length = 100
                            model = Sequential()
                            model.add(Embedding(len(events2), embedding_vecor_length, weights = [embeddings], input_length=3377, trainable = False))
                            model.add(LSTM(200))
                            model.add(Dense(2, activation='softmax'))
                            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                            print(model.summary())
                            model.fit(X_train1, y_train, validation_data=(X_test1, y_test), epochs=3, batch_size=64)

                            te_predict_Y = model.predict(X_test1)
                            te_predict_Y = te_predict_Y[:,1]


                            # ROC AUC
                            auc = metrics.roc_auc_score(y_test, te_predict_Y)
                            print('ROC AUC: %f' % auc)
                            AUC_hold.append(auc)



                            model.fit(X_train1, y_train, validation_data=(X_test2, y_test2), epochs=3, batch_size=64)

                            te_predict_Y2 = model.predict(X_test2)
                            te_predict_Y2 = te_predict_Y2[:,1]


                            # ROC AUC
                            auc2 = metrics.roc_auc_score(y_test2, te_predict_Y2)
                            print('ROC AUC: %f' % auc)
                            AUC_hold2.append(auc2)



                        else:
                            ratio = 'NA'
                            
                    else:
                        if len(np.unique(y_test)) == 2:
                            
                            y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
                            y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))

                            
                            ratios_prop.append(ratio)
                            print(ratio)


                            # truncate and pad input sequences
                            max_review_length = 3377
                            X_train1 = sequence.pad_sequences(X_train, maxlen=max_review_length)
                            X_test1 = sequence.pad_sequences(X_test, maxlen=max_review_length)

                            if analysis == 'split':

                                X_test2 = sequence.pad_sequences(X_test2, maxlen=max_review_length)

                            import sklearn.metrics as metrics
                            from sklearn.metrics import f1_score


                            # create the model
                            embedding_vecor_length = 100
                            model = Sequential()
                            model.add(Embedding(len(events2), embedding_vecor_length, weights = [embeddings], input_length=3377, trainable = False))
                            model.add(LSTM(200))
                            model.add(Dense(2, activation='softmax'))
                            model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
                            print(model.summary())
                            model.fit(X_train1, y_train, validation_data=(X_test1, y_test), epochs=3, batch_size=64)

                            te_predict_Y = model.predict(X_test1)
                            te_predict_Y = te_predict_Y[:,1]


                            # ROC AUC
                            auc = metrics.roc_auc_score(y_test, te_predict_Y)
                            print('ROC AUC: %f' % auc)
                            AUC_hold.append(auc)



            if condition == 'age':
                sentences, omit1, omit2, remove_patient_test,disease_prev = prep_data(valid_files)

                newsents = split_sentences(sentences)

                newsents2 = split_sentences2(sentences)

                target_loc = adjust_all(newsents2)

                disease_test, omit1 = adjust_exclusions(sentences, newsents,omit1)
                disease_test2, omit2 = adjust_exclusions(sentences, newsents,omit2)
  

                if test_type == 'complete':
                    p = 0

                if pred == 'lasso' or pred == 'DL' or pred == 'PDPS':

                    patient_seq_all_disease_test, exclude, ratio = patient_vec(valid_files, newsents,remove_patient_test,disease_prev, target_loc,omit1,p)
                    patient_seq_all_disease_test2, exclude2,ratio = patient_vec(valid_files, newsents,remove_patient_test,disease_prev, target_loc,omit2,p)
          
                    data_disease_test = pd.DataFrame(patient_seq_all_disease_test)
                    data_disease_test2 = pd.DataFrame(patient_seq_all_disease_test2)
             
                    X_disease_test, Y_disease_test = update_data(exclude,disease_test,data_disease_test)
                    X_disease_test2, Y_disease_test2 = update_data(exclude2,disease_test2,data_disease_test2)
               
                if pred == 'RNN':



                    X_test, exclude,ratio = RNN_index(valid_files, newsents, remove_patient_test, disease_prev,omit1, target_loc,p)
                    X_test2, exclude2,ratio = RNN_index(valid_files, newsents, remove_patient_test, disease_prev, omit2,target_loc,p)
                    X_test3, exclude3,ratio = RNN_index(valid_files, newsents, remove_patient_test,disease_prev,omit3, target_loc,p)
                    X_test4, exclude4,ratio = RNN_index(valid_files, newsents, remove_patient_test,disease_prev, omit4,target_loc,p)


                    y_test,emb_eddings = update_data_RNN(exclude,disease_test)
                    y_test2,emb_eddings2 = update_data_RNN(exclude2,disease_test2)
                    y_test3,emb_eddings3 = update_data_RNN(exclude3,disease_test3)
                    y_test4,emb_eddings4 = update_data_RNN(exclude4,disease_test4)

                    
                    
                if pred == 'lasso':

                    if len(np.unique(Y_disease_test)) == 2 and len(np.unique(Y_disease_test2)) == 2 and len(np.unique(Y_disease_test3)) == 2 and len(np.unique(Y_disease_test4)) == 2:


                        ratios_prop.append(ratio)
                        print(ratio)

                        searchCV_lasso = LogisticRegressionCV(Cs=list(np.power(10.0, np.arange(-10, 10))),penalty='l1',
                        cv=5,random_state=777,fit_intercept=True,solver='liblinear',max_iter = 10000,scoring="f1")
                        lasso_fit_disease = searchCV_lasso.fit(X_train_disease,Y_train_disease)

                        lasso_prob_disease = lasso_fit_disease.predict_proba(X_train_disease)[:,1]
                        lasso_prob_disease = lasso_fit_disease.predict_proba(X_disease_test)[:,1]
                        auc_disease_lasso = metrics.roc_auc_score(Y_disease_test, lasso_prob_disease)

                        lasso_prob_disease2 = lasso_fit_disease.predict_proba(X_disease_test2)[:,1]
                        auc_disease_lasso2 = metrics.roc_auc_score(Y_disease_test2, lasso_prob_disease2)

                        AUC_hold.append(auc_disease_lasso)
                        AUC_hold2.append(auc_disease_lasso2)
                  





                    else:
                        ratio = 'NA'
                        print(ratio)


                if pred == 'DL':
                    

                    if len(np.unique(Y_disease_test)) == 2 and len(np.unique(Y_disease_test2)) == 2 and len(np.unique(Y_disease_test3)) == 2 and len(np.unique(Y_disease_test4)) == 2:
                        ratios_prop.append(ratio)
                        print(ratio)


                        print("Reading files..."),
                        tr_dataframe = data_disease
                        te_dataframe = data_disease_test
                        te_dataframe2 = data_disease_test2
                        te_dataframe3 = data_disease_test3
                        te_dataframe4 = data_disease_test4


                        print("Fineshed\n")

                        print("Compressing data..."),
                        tr_X = tr_dataframe.iloc[:, :-1].values
                        tr_Y = tr_dataframe.iloc[:, -1].values
                        te_X = te_dataframe.iloc[:, :-1].values
                        te_Y = te_dataframe.iloc[:, -1].values
                        te_X2 = te_dataframe2.iloc[:, :-1].values
                        te_Y2 = te_dataframe2.iloc[:, -1].values
                        te_X3 = te_dataframe3.iloc[:, :-1].values
                        te_Y3 = te_dataframe3.iloc[:, -1].values
                        te_X4 = te_dataframe4.iloc[:, :-1].values
                        te_Y4 = te_dataframe4.iloc[:, -1].values





                        # model_DL = KerasRegressor(build_fn = baseline_model(units = Grid.best_params_['units'], act = Grid.best_params_['act']), nb_epoch=2000, batch_size=300,verbose = 0)

                        model_DL = Sequential()
                        model_DL.add(Dense(55,activation = 'relu'))
                        model_DL.add(Dense(20,activation='relu'))
                        model_DL.add(Dense(1,activation = 'sigmoid'))
                        model_DL.compile(loss='mean_squared_error', optimizer='adam')



                        # In[ ]:




                        # In[32]:


                        model_DL.fit(tr_X,tr_Y,batch_size = 100, epochs=70, verbose = 0)
                        tr_predict_Y = model_DL.predict(tr_X)
                        te_predict_Y = model_DL.predict(te_X)    






                        # ROC AUC
                        auc = metrics.roc_auc_score(te_Y, te_predict_Y)
                        print('ROC AUC: %f' % auc)





                        te_predict_Y2 = model_DL.predict(te_X2) 

                         # ROC AUC
                        auc2 = metrics.roc_auc_score(te_Y2, te_predict_Y2)
                        print('ROC AUC2: %f' % auc2)

                        te_predict_Y3 = model_DL.predict(te_X3) 

                         # ROC AUC
                        auc3 = metrics.roc_auc_score(te_Y3, te_predict_Y3)
                        print('ROC AUC3: %f' % auc3)

                        te_predict_Y4 = model_DL.predict(te_X4) 

                         # ROC AUC
                        auc4 = metrics.roc_auc_score(te_Y4, te_predict_Y4)
                        print('ROC AUC4: %f' % auc4)


                        AUC.append(auc)
                        AUC2.append(auc2)
                        AUC3.append(auc3)
                        AUC4.append(auc4)



                    else:
                        ratio = 'NA'



                if pred == 'RNN':

                    if len(np.unique(y_test)) == 2 and len(np.unique(y_test2)) == 2:
                        ratios_prop.append(ratio)
                        print(ratio)

                        # truncate and pad input sequences
                        max_review_length = 3377
                        X_train1 = sequence.pad_sequences(X_train, maxlen=max_review_length)
                        X_test1 = sequence.pad_sequences(X_test, maxlen=max_review_length)
                        X_test2 = sequence.pad_sequences(X_test2, maxlen=max_review_length)

                        import sklearn.metrics as metrics
                        from sklearn.metrics import f1_score


                        # create the model
                        embedding_vecor_length = 100
                        model = Sequential()
                        model.add(Embedding(len(events2), embedding_vecor_length, weights = [embeddings], input_length=3377, trainable = False))
                        model.add(LSTM(100))
                        model.add(Dense(1, activation='sigmoid'))
                        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                        print(model.summary())
                        model.fit(X_train1, y_train, validation_data=(X_test1, y_test), epochs=3, batch_size=64)

                        te_predict_Y = model.predict(X_test1)


                        # ROC AUC
                        auc = metrics.roc_auc_score(y_test, te_predict_Y)
                        print('ROC AUC: %f' % auc)
                        AUC.append(auc)


                        model.fit(X_train1, y_train, validation_data=(X_test2, y_test2), epochs=3, batch_size=64)

                        te_predict_Y2 = model.predict(X_test2)


                        # ROC AUC
                        auc2 = metrics.roc_auc_score(y_test2, te_predict_Y2)
                        print('ROC AUC: %f' % auc2)
                        AUC2.append(auc2)


                        model.fit(X_train1, y_train, validation_data=(X_test3, y_test3), epochs=3, batch_size=64)

                        te_predict_Y3 = model.predict(X_test3)


                        # ROC AUC
                        auc3 = metrics.roc_auc_score(y_test3, te_predict_Y3)
                        print('ROC AUC3: %f' % auc3)
                        AUC3.append(auc3)

                        model.fit(X_train1, y_train, validation_data=(X_test4, y_test4), epochs=3, batch_size=64)

                        te_predict_Y4 = model.predict(X_test4)


                        # ROC AUC
                        auc4 = metrics.roc_auc_score(y_test4, te_predict_Y4)
                        print('ROC AUC4: %f' % auc4)
                        AUC4.append(auc4)



                    else:
                        ratio = 'NA'


                if pred == 'PDPS':

                    if len(np.unique(Y_disease_test)) == 2 and len(np.unique(Y_disease_test2)) == 2:
                        ratios_prop.append(ratio)
                        print(ratio)
                        cos_sim_test = []
                        f1_calc = []
                        for count in range(len(patient_seq_all_disease_test)): 
                            a = patient_seq_all_disease_test[count]
                            b = model.wv.word_vec(passed[q],use_norm=norm)
                            norma = np.sqrt(a.dot(a))
                            normb = np.sqrt(b.dot(b))
                            cos_sim_test.append(dot(a, b)/(norma*normb))

                        aucc=(metrics.roc_auc_score(Y_disease_test, cos_sim_test))
                        AUC.append(metrics.roc_auc_score(Y_disease_test, cos_sim_test))
                        print('AUC disease Dys:,',aucc)

                        cos_sim_test2 = []
                        f1_calc = []
                        for count in range(len(patient_seq_all_disease_test2)): 
                            a = patient_seq_all_disease_test2[count]
                            b = model.wv.word_vec(passed[q],use_norm=norm)
                            norma = np.sqrt(a.dot(a))
                            normb = np.sqrt(b.dot(b))
                            cos_sim_test2.append(dot(a, b)/(norma*normb))

                        aucc2=(metrics.roc_auc_score(Y_disease_test2, cos_sim_test2))
                        AUC2.append(metrics.roc_auc_score(Y_disease_test2, cos_sim_test2))
                        print('AUC disease Dys2:,',aucc2)

                        cos_sim_test3 = []
                        f1_calc = []
                        for count in range(len(patient_seq_all_disease_test3)): 
                            a = patient_seq_all_disease_test3[count]
                            b = model.wv.word_vec(passed[q],use_norm=norm)
                            norma = np.sqrt(a.dot(a))
                            normb = np.sqrt(b.dot(b))
                            cos_sim_test3.append(dot(a, b)/(norma*normb))

                        aucc3=(metrics.roc_auc_score(Y_disease_test3, cos_sim_test3))
                        AUC3.append(metrics.roc_auc_score(Y_disease_test3, cos_sim_test3))
                        print('AUC disease Dys:,',aucc3)


                        cos_sim_test4 = []
                        f1_calc = []
                        for count in range(len(patient_seq_all_disease_test4)): 
                            a = patient_seq_all_disease_test4[count]
                            b = model.wv.word_vec(passed[q],use_norm=norm)
                            norma = np.sqrt(a.dot(a))
                            normb = np.sqrt(b.dot(b))
                            cos_sim_test4.append(dot(a, b)/(norma*normb))

                        aucc4=(metrics.roc_auc_score(Y_disease_test4, cos_sim_test4))
                        AUC4.append(metrics.roc_auc_score(Y_disease_test4, cos_sim_test4))
                        print('AUC disease Dys:,',aucc4)



                    else:
                        ratio = 'NA'
                        
        AUC.append(np.mean(AUC_hold))
        ratios_prop.append(np.mean(ratios_hold))
        ratios_prop2.append(np.mean(ratios_hold2))
        if analysis == 'split':
            if condition == 'insurance' or condition == 'gender' or condition == 'age2' or condition == 'race':
                AUC2.append(np.mean(AUC_hold2))
            else:
                AUC2.append(np.mean(AUC_hold2))
                AUC3.append(np.mean(AUC_hold3))
                AUC4.append(np.mean(AUC_hold4))


    if condition == 'insurance' or condition == 'gender' or condition == 'age2' or condition == 'race':

        dataset_auc.append(AUC)
        dataset_auc2.append(AUC2)
        ratios_dis.append(ratios_prop)
        ratios_dis2.append(ratios_prop2)

    if condition == 'age':
        dataset_auc.append(AUC)
        dataset_auc2.append(AUC2)
        dataset_auc3.append(AUC3)
        dataset_auc4.append(AUC4)
        ratios_dis.append(ratios_prop)

if condition == 'insurance' or condition == 'gender' or condition == 'age2' or condition == 'race': 

    from pandas import DataFrame
    df_auc = DataFrame(dataset_auc)
    df_auc.to_csv(path_auc)
    df_auc2 = DataFrame(dataset_auc2)
    df_auc2.to_csv(path_auc2)
    df_ratio = DataFrame(ratios_dis)
    df_ratio.to_csv(path_ratios)
    df_ratio2 = DataFrame(ratios_dis2)
    df_ratio2.to_csv(path_ratios2)
if condition == 'age':
    from pandas import DataFrame
    df_auc = DataFrame(dataset_auc)
    df_auc.to_csv(path_auc)
    df_auc2 = DataFrame(dataset_auc2)
    df_auc2.to_csv(path_auc2)
    df_auc3 = DataFrame(dataset_auc3)
    df_auc3.to_csv(path_auc3)
    df_auc4 = DataFrame(dataset_auc4)
    df_auc4.to_csv(path_auc4)
    df_ratio = DataFrame(ratios_dis)
    df_ratio.to_csv(path_ratios)



