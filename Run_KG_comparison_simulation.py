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
import os
import pickle

def run(data_path, kg_path, model_type = 'DL',test_type = 'incomplete'):

lib_path = os.path.abspath(os.path.join('..', 'lib'))
sys.path.append(lib_path)

#ds = "mimic"
#data_path = "/Users/emily/Documents/SequentialPhenotypePredictor-master/Data/mimic_seq/"
#data_path = '/home/egetzen/mimic_seq/'
window=10
size=100
decay=5
skipgram=1
norm=False


## list for files
train_files = []
valid_files = []
full_data_files = []
for i in range(10):
    full_data_files.append(data_path + 'test_'+str(i))

## train files
events_files = []
for i in range(7):
    events_files.append(data_path + 'test_'+str(i))

## Create sentence representation from the codified data   
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

## Get unique medical concepts
import itertools
flat_sent = list(itertools.chain(*sentences))
events2 = np.unique(flat_sent)
events2 = np.ndarray.tolist(events2)
len(events2)


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




## Diseases to evaluate
passed = ['d_250','d_585','d_428','d_403','d_272']



pred = model_type
analysis = 'not_split'
condition = 'gender'
file_path = data_path + 'results/'

## Read in knowledge graph
dice_mat = pd.read_csv(kg_path,header=None)
    
if analysis != 'split':
    if test_type == 'complete':
        if pred == 'lasso':
            path_auc = file_path + 'KG_lasso_aucc1'
            path_auc2 = file_path + 'KG_lasso_aucc2'
            path_ratios = file_path + 'KG_lasso_ratioc'
           
            
        if pred == 'DL':
            path_auc = file_path + 'KG_DL_aucc1'
            path_auc2 = file_path + 'KG_DL_aucc2'
            path_ratios = file_path + 'KG_DL_ratioc'
            
    if test_type == 'incomplete':
        if pred == 'lasso':
            path_auc = file_path + 'KG_lasso_auci1'
            path_auc2 = file_path + 'KG_lasso_auci2'
            path_ratios = file_path + 'KG_lasso_ratioi'
       
            
        if pred == 'DL':
            path_auc = file_path + 'KG_DL_auci1'
            path_auc2 = file_path + 'KG_DL_auci2'
            path_ratios = file_path + 'KG_DL_ratioi'
            
dat = list(range(17,100,1))


dataset_auc = []
dataset_auc2 = []
dataset_auc3 = []
dataset_auc4 = []
ratios_dis = []

for q in range(len(passed)):  
    AUC = []
    AUC2 = []
    AUC3 = []
    AUC4 = []
    ratios_prop = []
    ratios_prop2 = []

    ## Different amounts of missingness, these proportions lead to the same total amount missing without KG
    prop = [0,.055,.32,.59,.75]

    for h in range(len(prop)):
        p = prop[h]
        AUC_hold = []
        AUC_hold2 = []
        AUC_hold3 = []
        AUC_hold4 = []
        ratios_hold = []
        ratios_hold2 = []
        for w in range(200): ## 200 iterations and average results 
            valid_files = []          
            p = prop[h]
            iters = w
            random.seed(range(200)[iters])
            
            
            train_files = random.sample(full_data_files,7)
            for i in range(len(full_data_files)):
                if full_data_files[i] not in train_files:
                    valid_files.append(full_data_files[i])
                    
            print(train_files)
            print(valid_files)
            
            # In[5]:

            ## Get sentence structure for train files
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
                disease_prev = []
                omit1 = []
                count = -1
                omit2 = []
                omit3 = []
                omit4 = []


                for i in data_files:
                    with open(i) as f:
                        for s in f:
                            count = count+1
                            age = float(s.split("|")[1].split(" ")[3].replace(",",r""))
                            gen = int(s.split("|")[1].split(" ")[1].replace(",",r""))
                            insurance = s.split("|")[1].split(" ")[5].replace(",",r"")
                            if age <= 5:
                                age_dyn.append(0)
                            if age > 5 and age <= 50:
                                age_dyn.append(1)
                            if age > 50:
                                age_dyn.append(2)


                            if insurance == '"Medicare"' or insurance == '"Medicaid"' or insurance == '"Government"':
                                Xins.append(-2.24)
                            else:
                                Xins.append(.07)


                            gender.append(int(s.split("|")[1].split(" ")[1].replace(",",r"")))
                            if gen == 0:
                                Xgen.append(-3.6)
                            else:
                                Xgen.append(.13)
                            Xage.append(((age-np.mean(dat))/(.5*np.std(dat)))-.9)

                            if data_files[0][74:100]=='test_0' or analysis != 'split':
                                if age <= 5:
                                    omit1.append(1)
                                else:
                                    omit1.append(0)
                            else:
                                if analysis == 'split':
                                    if condition == 'gender':                                
                                        if age <= 5 or gen== 1:
                                            omit1.append(1)
                                        else:
                                            omit1.append(0)
                                        if age <= 5 or gen == 0:
                                            omit2.append(1)                                
                                        else:
                                            omit2.append(0)
                                    if condition == 'insurance':                                
                                        if age <= 5 or insurance != '"Medicare"' and insurance != '"Medicaid"' and insurance != '"Government"':
                                            omit1.append(1)
                                        else:
                                            omit1.append(0)
                                        if age <= 5 or insurance == '"Medicare"' or insurance == '"Medicaid"' or insurance == '"Government"':
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
                            if passed[q] in prev_diags: # if disease exists in the previous diagnosis
                                disease_prev.append(1)
                            else:
                                disease_prev.append(0)
                            sentences.append(s.split("|")[2].split(" ") +
                                                 s.split("|")[3].replace("\n", "").split(" "))
                            B1 = -0.62
                            B2 = -1.2
                            B3 = -.7

                        p_select = [1 for e in range(len(age_dyn))]
                        

                     
                        
                        
                        ran = range(200)[iters]
                        random.seed(ran)
                        np.random.seed(ran)
                        tf.random.set_seed(ran)
                        
                        
                        remove_patient_data = [int(np.random.binomial(1,p_select[e],1)) for e in range(len(age_dyn))]



                return sentences, omit1, omit2, remove_patient_data, disease_prev
            sentences, omit1, omit2,remove_patient_train,  disease_prev = prep_data(train_files,iters)



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

            
            ## Within each patient, split medical record into individual visits  
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


        ##Go through and remove medical events up to a desired proportion using the knowledge graph


            def patient_vec(data_files, newsents, remove_patient_data, disease_prev, target,omit,p,iters):
                
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
                            if omit[count]==1 or newsents[count]==[] or newsents[count][0:-1]==[]:
                                exclude.append(1)
                            else:
                                if disease_prev[count] == 1:  
                                    visits = newsents[count][:target[count]]

                                    if remove_patient_data[count]==1:
                                        before.append(len(list(np.concatenate(visits))))
                                        new_visits_all = []
                                        test = np.random.binomial(1,0,len(visits))
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
                                            if new_visits_all == []:
                                                new_visits_all = visits 
                                            visits = new_visits_all 
                                            if visits == []:
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
                                    visits = newsents[count][0:-1]  

                                    if remove_patient_data[count]==1:
                                        before.append(len(list(np.concatenate(visits))))
                                        new_visits_all = []
                                        test = np.random.binomial(1,0,len(visits))
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
                                            if new_visits_all == []:
                                                new_visits_all = visits 
                                            visits = new_visits_all 
                                            if visits == []:
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

                if data_files == train_files:
                    ratio = 1-sum(after)/sum(before)

                return new_sentences, exclude, ratio
            
            
            

            ## Update labels for presence of target diagnosis
            def update_data(exclude,disease_data,data_disease):                
                disease_final = []
                for i in range(0,len(exclude)):
                    if exclude[i] == 0:
                        disease_final.append(disease_data[i])

                data_disease[passed[q]] = disease_final

                X_data_disease=data_disease.iloc[:,0:100]
                Y_data_disease =data_disease.iloc[:,100]


                return X_data_disease, Y_data_disease

        
            
            new_sentences, exclude, ratio = patient_vec(train_files, newsents, remove_patient_train, disease_prev,target_loc,omit1,p,iters)

            import os
            os.environ['PYTHONHASHSEED'] = '0'
            os.environ['CUDA_VISIBLE_DEVICES']='-1'
            os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'

            ran = range(200)[iters]
            random.seed(ran)
            np.random.seed(ran)
            tf.random.set_seed(ran)

            ## Using the adjusted patient record, get a word2vec embedding of each remaining medical concept and create a patient vector representation via temporal averaging
            print('fitting word2vec model')
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


                X_train, exclude, ratio = patient_vec(train_files, newsents, remove_patient_train, disease_prev,target_loc,omit1,p,iters)
                
                dict_events = {k: v for v, k in enumerate(events2)}
                X_train_ind = []
                for i in range(len(X_train)):
                    X_train_ind.append([dict_events[X_train[i][j]] for j in range(len(X_train[i])) if X_train[i][j] in events2])

                X_train = X_train_ind
 

                y_train = update_data_RNN(exclude, disease_train)
                
     

                sentences, omit1, omit2, remove_patient_test, disease_prev = prep_data(valid_files,iters)

                newsents = split_sentences(sentences)

                newsents2 = split_sentences2(sentences)

                target_loc = adjust_all(newsents2)

                disease_test, omit1 = adjust_exclusions(sentences, newsents,omit1)

               
                if test_type == 'complete':
                    p = 0

                if pred == 'lasso' or pred == 'DL' or pred == 'PDPS':

                    new_sentences, exclude, ratio2 = patient_vec(valid_files, newsents,remove_patient_test,disease_prev, target_loc,omit1,p,iters)

                                
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


          


                    data_disease_test = pd.DataFrame(patient_seq_all_disease_test)

                    

                    X_disease_test, Y_disease_test = update_data(exclude,disease_test,data_disease_test)


                if pred == 'lasso' or pred == 'DL' or pred == 'PDPS':


                    if pred == 'lasso':

                        if analysis == 'split':

                            if len(np.unique(Y_disease_test)) == 2 and len(np.unique(Y_disease_test2)) == 2:

                                ratios_hold.append(ratio)
                                print(ratio)
                                
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



                        
        AUC.append(np.mean(AUC_hold))
        ratios_prop.append(np.mean(ratios_hold))
        if analysis == 'split':
            if condition == 'insurance' or condition == 'gender' or condition == 'age2':
                AUC2.append(np.mean(AUC_hold2))
            else:
                AUC2.append(np.mean(AUC_hold2))
                AUC3.append(np.mean(AUC_hold3))
                AUC4.append(np.mean(AUC_hold4))


    if condition == 'insurance' or condition == 'gender' or condition == 'age2':

        dataset_auc.append(AUC)
        dataset_auc2.append(AUC2)
        ratios_dis.append(ratios_prop)

    if condition == 'age':
        dataset_auc.append(AUC)
        dataset_auc2.append(AUC2)
        dataset_auc3.append(AUC3)
        dataset_auc4.append(AUC4)
        ratios_dis.append(ratios_prop)

if condition == 'insurance' or condition == 'gender' or condition == 'age2': 

    from pandas import DataFrame
    df_auc = DataFrame(dataset_auc)
    df_auc.to_csv(path_auc)
    df_auc2 = DataFrame(dataset_auc2)
    df_auc2.to_csv(path_auc2)
    df_ratio = DataFrame(ratios_dis)
    df_ratio.to_csv(path_ratios)

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

