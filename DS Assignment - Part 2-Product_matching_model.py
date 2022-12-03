#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libraries
import time
import re
import pandas as pd
import numpy as np
import jellyfish as jf
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


# In[2]:


#Load the synthetic data prepared from product_match data
df = pd.read_csv('product_matching_synthetic.csv')
df.head()


# In[3]:


#Create function for calculation of length of characters for matching
def matching_numbers(external_name, internal_name):

    external_numbers = set(re.findall(r'[0-9]+', external_name))
    internal_numbers = set(re.findall(r'[0-9]+', internal_name))    
    union = external_numbers.union(internal_numbers)
    intersection = external_numbers.intersection(internal_numbers)

    if len(external_numbers)==0 and len(internal_numbers) == 0:
        return 1
    else:
        return (len(intersection)/ len(union))


# In[4]:


#Create function for character similarity between the product names
def engineer_features(df):

    df['Product name in Amazon'] = df['Product name in Amazon'].str.lower()
    df['Product name in Flipkart'] = df['Product name in Flipkart'].str.lower()

    df['levenshtein_distance'] = df.apply(
    lambda x: jf.levenshtein_distance(x['Product name in Flipkart'], 
                                      x['Product name in Amazon']), axis=1)

    df['damerau_levenshtein_distance'] = df.apply(
    lambda x: jf.damerau_levenshtein_distance(x['Product name in Flipkart'], 
                                              x['Product name in Amazon']), axis=1)

    df['hamming_distance'] = df.apply(
    lambda x: jf.hamming_distance(x['Product name in Flipkart'], 
                                  x['Product name in Amazon']), axis=1)

    df['jaro_similarity'] = df.apply(
    lambda x: jf.jaro_similarity(x['Product name in Flipkart'], 
                                  x['Product name in Amazon']), axis=1)

    df['jaro_winkler_similarity'] = df.apply(
    lambda x: jf.jaro_winkler_similarity(x['Product name in Flipkart'], 
                                         x['Product name in Amazon']), axis=1)

    df['match_rating_comparison'] = df.apply(
    lambda x: jf.match_rating_comparison(x['Product name in Flipkart'], 
                                         x['Product name in Amazon']), axis=1).fillna(0).astype(int)

    df['ratio'] = df.apply(
    lambda x: fuzz.ratio(x['Product name in Flipkart'], 
                         x['Product name in Amazon']), axis=1)

    df['partial_ratio'] = df.apply(
    lambda x: fuzz.partial_ratio(x['Product name in Flipkart'], 
                                 x['Product name in Amazon']), axis=1)

    df['token_sort_ratio'] = df.apply(
    lambda x: fuzz.token_sort_ratio(x['Product name in Flipkart'], 
                                    x['Product name in Amazon']), axis=1)

    df['token_set_ratio'] = df.apply(
    lambda x: fuzz.token_set_ratio(x['Product name in Flipkart'], 
                                   x['Product name in Amazon']), axis=1)

    df['w_ratio'] = df.apply(
    lambda x: fuzz.WRatio(x['Product name in Flipkart'], 
                          x['Product name in Amazon']), axis=1)

    df['uq_ratio'] = df.apply(
    lambda x: fuzz.UQRatio(x['Product name in Flipkart'], 
                          x['Product name in Amazon']), axis=1)

    df['q_ratio'] = df.apply(
    lambda x: fuzz.QRatio(x['Product name in Flipkart'], 
                          x['Product name in Amazon']), axis=1)    

    df['matching_numbers'] = df.apply(
    lambda x: matching_numbers(x['Product name in Flipkart'], 
                               x['Product name in Amazon']), axis=1)

    df['matching_numbers_log'] = (df['matching_numbers']+1).apply(np.log)

    df['log_fuzz_score'] = (df['ratio'] + df['partial_ratio'] + 
                            df['token_sort_ratio'] + df['token_set_ratio']).apply(np.log)

    df['log_fuzz_score_numbers'] = df['log_fuzz_score'] + (df['matching_numbers']).apply(np.log)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(value=0, inplace=True)

    return df


# In[5]:


products = engineer_features(df)
products


# In[6]:


#Apply correlation to find the best calculation ratio among various functions
products[products.columns[1:]].corr()['match'][:].sort_values(ascending=False)


# In[7]:


#Splitting data for training and test set
X = products[['levenshtein_distance', 'damerau_levenshtein_distance', 'hamming_distance',
       'jaro_similarity','jaro_winkler_similarity','matching_numbers_log',
       'matching_numbers','token_set_ratio','token_sort_ratio','partial_ratio',
       'ratio','log_fuzz_score','log_fuzz_score_numbers','match_rating_comparison',
       'q_ratio','uq_ratio','w_ratio']].values
y = products['match'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# In[8]:


#Create confusion matrix to find the best classifier model
def get_confusion_matrix_values(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

classifiers = {
    "DummyClassifier_stratified":DummyClassifier(strategy='stratified', random_state=0),    
    "KNeighborsClassifier":KNeighborsClassifier(3),
    "XGBClassifier":XGBClassifier(n_estimators=1000, learning_rate=0.1),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "RandomForestClassifier":RandomForestClassifier(),
    "AdaBoostClassifier":AdaBoostClassifier(),
    "GradientBoostingClassifier":GradientBoostingClassifier(),
    "Perceptron": Perceptron(max_iter=40, eta0=0.1, random_state=1),
    "MLP": MLPClassifier(),
    "XGBClassifer tuned": XGBClassifier(colsample_bytree=0.8,
                      gamma=0.9,
                      max_depth=20,
                      min_child_weight=1,
                      scale_pos_weight=12,
                      subsample=0.9,
                      n_estimators=50, 
                      learning_rate=0.1)
}

df_results = pd.DataFrame(columns=['model', 'accuracy', 'mae', 'precision',
                                   'recall','f1','roc','run_time','tp','fp',
                                   'tn','fn'])

for key in classifiers:

    start_time = time.time()
    classifier = classifiers[key]
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred, zero_division=0)
    run_time = format(round((time.time() - start_time)/60,2))
    tp, fp, fn, tn = get_confusion_matrix_values(y_test, y_pred)

    row = {'model': key,
           'accuracy': accuracy,
           'mae': mae,
           'precision': precision,
           'recall': recall,
           'f1': f1,
           'roc': roc,
           'run_time': run_time,
           'tp': tp,
           'fp': fp,
           'tn': tn,
           'fn': fn,
          }
    df_results = df_results.append(row, ignore_index=True)


# In[9]:


df_results.head(10)


# In[10]:


#Best Parameter tuning calculation(sclae_pos_weight)
def get_scale_pos_weight(target, square_root=False, gridsearch=False):
    import math

    scale_pos_weight = round((len(target) - sum(target)) / sum(target))

    if square_root:
        scale_pos_weight = round(math.sqrt(scale_pos_weight))

    if gridsearch:
        scale_pos_weight = [scale_pos_weight-2, scale_pos_weight-1, scale_pos_weight, 
                            scale_pos_weight+1, scale_pos_weight+2]

    return scale_pos_weight


# In[11]:


scale_pos_weight = get_scale_pos_weight(products['match'], square_root=False, gridsearch=True)
scale_pos_weight


# In[12]:


#Optimum parameter calculation using gridsearchCV  
n_estimators = [50]
learning_rate = [0.1]
max_depth = [5, 10, 20]
min_child_weight = [1, 2]
scale_pos_weight = [4,5,6,7,8]
gamma = [0.9, 1.0]
subsample = [0.9]
colsample_bytree = [0.8, 1.0]

start = time.perf_counter()

param_grid = dict(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                scale_pos_weight=scale_pos_weight,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
)

model = XGBClassifier(random_state=0)

grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           )

print('Running GridSearchCV...')
best_model = grid_search.fit(X_train, y_train)
best_score = round(best_model.score(X_test, y_test), 4)
best_params = best_model.best_params_

print('Score:', best_score)
print('Optimum parameters', best_params)

finish = time.perf_counter()
run_time = (finish - start / 60)
print(f"Completed task in {run_time:0.4f} minutes")


# In[13]:


#Model creation using XGBoost classifier
model = XGBClassifier(colsample_bytree=0.8,
                      gamma=0.9,
                      max_depth=10,
                      min_child_weight=1,
                      scale_pos_weight=5,
                      subsample=0.9,
                      n_estimators=50, 
                      learning_rate=0.1)
model = classifier.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[14]:


print(classification_report(y_test, y_pred, labels=[1, 0], 
                            target_names=['match', 'not match']))


# In[15]:


#Prediction result cross check
results = pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})
results['result'] = np.where(results['predictions']==results['actual'], 1, 0)
results.head(20)


# In[24]:


def get_closest_matches(external_name):

    unique_internal_names = products['Product name in Amazon'].unique().tolist()
    closest_matches = process.extract(external_name, 
                  unique_internal_names, 
                  scorer=fuzz.token_set_ratio)

    return closest_matches


# In[25]:


def prepare_data(external_name):

    closest_matches = get_closest_matches(external_name)

    df = pd.DataFrame(columns=['external_name', 'internal_name'])

    for match in closest_matches:
        row = {'external_name': external_name, 'internal_name': match[1]}
        df = df.append(row, ignore_index=True)

    return df


# In[26]:


closest_data = prepare_data("alisha solid women's cycling shorts")
closest_data.head()


# In[ ]:




