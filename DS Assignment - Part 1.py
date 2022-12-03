#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns


# In[3]:


#Loading the dataset
data = pd.read_excel("DS - Assignment Part 1 data set.xlsx")


# In[5]:


#Descriptive analysis
data.describe()


# In[6]:


#Check for null values
data.isnull().sum()


# In[8]:


#Checkfing correlation
sns.heatmap(data.corr(),annot = True,cmap='magma')
plt.show()


# In[9]:


#Creation of new column from the existing(Since the data contain house price only for unit area, I have calculated the original house price)
data['House_price'] = data['House size (sqft)']*data['House price of unit area']


# In[10]:


data.head()


# In[11]:


#Correlate house price with other variables
cor = data.corr()
corr_house_price = cor['House_price']
print(corr_house_price)


# In[12]:


#Heatmap creation on correlation data
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,10))
sns.heatmap(cor, annot=True, cmap='magma')
plt.show()


# In[13]:


#Dropping the unnecessary columns from the dataset
data = data.drop(['Transaction date','latitude','longitude'],axis=1)


# In[14]:


#Calculating inter-quantile range for removing outliers
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
IQR = q3 - q1
print(IQR)


# In[16]:


data_res = data[~((data< (q1 - 1.5 * IQR)) | (data > (q3 + 1.5 * IQR))).any(axis=1)]
data_res.shape


# In[18]:


#Import other libraries for analysis
import scipy
from scipy.stats import norm
from scipy import stats
#Checking for normality
sns.distplot(data['House_price'],fit=norm)
fig = plt.figure()
res = stats.probplot(data['House_price'],plot=plt)


# In[19]:


#Splitting the target and independent variables
X = data.iloc[:,:-1]
y = data.iloc[:,-1]


# In[20]:


#Feature selection using RFE function
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
rfe_reg = LinearRegression()
rfe = RFE(rfe_reg,n_features_to_select = 2)#random number of features = 2
X_rfe = rfe.fit_transform(X,y)
rfe_reg.fit(X,y)
print(rfe.support_)
print(rfe.ranking_)


# In[22]:


#Finding the optimum features
from sklearn.model_selection import train_test_split
fea_list = np.arange(1,7)
high_score = 0
fea = 0
score_list = []
for n in range(len(fea_list)):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=0)
    model = LinearRegression()
    rfe = RFE(model,n_features_to_select = fea_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        fea = fea_list[n]
print("optimum no. of features: %d" %fea)
print("score with %d features: %f" %(fea,high_score))


# In[23]:


rfe = RFE(rfe_reg,n_features_to_select = 5)
X_rfe = rfe.fit_transform(X,y)
rfe_reg.fit(X,y)
print(rfe.support_)
print(rfe.ranking_)


# In[24]:


RFE_features = X.iloc[:,[0 & 1,2,3,4,5]]


# In[25]:


RFE_features


# In[26]:


#Linear regression on the final data
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)


# In[27]:


print("score: ",lin_reg.score(X_train,y_train))
print("Model slope:    ", lin_reg.coef_)
print("Model intercept:", lin_reg.intercept_)


# In[28]:


#Prediction using linear regression
y_pred = lin_reg.predict(X_test)
print("score: ",lin_reg.score(X_test,y_test))
print("Model slope:    ", lin_reg.coef_)
print("Model intercept:", lin_reg.intercept_)


# In[29]:


#Gradient boosting regression on the final data
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators=500,max_depth=3,
                                min_samples_split=2,learning_rate=0.1,loss='ls')
est.fit(X_train,y_train)
est.score(X_test,y_test)


# In[30]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth = 2,random_state=0)
dt.fit(X_train,y_train)
print('Train score for decision tree regression:',dt.score(X_train,y_train))
print('Test score for decision tree regression:',dt.score(X_test,y_test))


# In[41]:


#Prediction using Gradient boosting regression
y_pred_dt = dt.predict(X_test)
y_pred_dt


# In[31]:


#Decision tree regression on the final data
from sklearn.model_selection import GridSearchCV
tree = DecisionTreeRegressor()
param_grid = {'max_depth': [1, 5, 10, 25, 50]}
print("Parameter grid:\n{}".format(param_grid))
grid_search_dr = GridSearchCV(tree, param_grid, cv=5,scoring="r2", return_train_score=True )
grid_search_dr.fit(X_train, y_train)
print("Best parameter:{}".format(grid_search_dr.best_params_))
print("Best cv score:{:.5f}".format(grid_search_dr.best_score_))


# In[42]:


#Prediction using Decision tree regression
y_pred_grid = grid_search_dr.predict(X_test)
y_pred_grid


# In[43]:


y_test


# In[33]:


#Bagging Decision tree regression on the final data
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
dr_best = DecisionTreeRegressor(random_state=0,max_depth=25)
bag_dt = BaggingRegressor(dr_best,bootstrap=True,random_state=0)
bag_dt.fit(X_train,y_train)
print('Train score for Bagging decision tree regression:',bag_dt.score(X_train,y_train))
print('Test score for Bagging decision tree regression:',bag_dt.score(X_test,y_test))


# In[37]:


X_test


# In[40]:


#Prediction using Bagging Decision tree regression
y_pred_bag_dt = bag_dt.predict(X_test)
y_pred_bag_dt

