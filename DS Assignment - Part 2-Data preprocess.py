#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading the data from flipkart sample file
df_1 = pd.read_csv("flipkart_com-ecommerce_sample.csv")
df_1.head()


# In[4]:


#Data manipulation for creating category column
category_fk = df_1['product_category_tree'].str.split('>>', 1).str[0]
category_fk = category_fk.str.replace(r'[\W\s]','')
category_fk


# In[5]:


df_1.isna().sum()


# In[6]:


#Filtering the required column for flipkart data(fk)
fk = df_1[['product_name','retail_price','discounted_price','pid']]
fk.columns = ['Product name in Flipkart','Retail price in Flipkart','Discounted price in Flipkart','Product id']


# In[7]:


#Adding category column to the existing fk data
fk['Product category in Flipkart'] = category_fk


# In[8]:


fk


# In[9]:


#Dropping na values from the dataset
fk = fk.dropna(axis=0)


# In[10]:


fk.head()


# In[11]:


#Loading the data from amazon sample file
df_2 = pd.read_csv("amz_com-ecommerce_sample.csv",encoding= 'unicode_escape')
df_2.head()


# In[12]:


#Redoing the manipulation for category column from amazon dataset
category_amz = df_2['product_category_tree'].str.split('>>', 1).str[0]
category_amz = category_amz.str.replace(r'[\W\s]','')
category_amz


# In[13]:


#Filtering the required columns from the amazon data
amz = df_2[['product_name','retail_price','discounted_price','pid']]
amz.columns = ['Product name in Amazon','Retail price in Amazon','Discounted price in Amazon','Product id']


# In[14]:


#Adding category column to the amazon data
amz['Product category in Amazon'] = category_amz


# In[15]:


#Dropping the na values
amz.isna().sum()


# In[16]:


#Joining the data using product id from the two datasets
product_matched = pd.merge(fk,amz,on='Product id',how = 'inner')
product_matched.head()


# In[17]:


#Creating addition column of match as 1 to identify all matched datas
product_matched['match'] = 1


# In[22]:


#Creating function for synthetic data preparation
def create_synthetic_data(df, iterations):
    """Creates synthetic training data from the correctly matched
    data by grouping on the cluster_label column and reshuffling
    the Product name in Amazon to create data that contain incorrect matches.
    """

    df_output = df

    i = 1
    while i <= iterations:

        # Create synthetic data by shuffling the column using a groupby
        df_s = df[['Product name in Flipkart','Retail price in Flipkart','Discounted price in Flipkart','Product name in Amazon','Retail price in Amazon','Discounted price in Amazon','Product category in Flipkart']].copy()
        df_s['shuffled_Product name in Amazon'] = df_s['Product name in Amazon']
        df_s['shuffled_Product name in Amazon'] = df_s.groupby('Product category in Flipkart')['Product name in Amazon'].transform(np.random.permutation)

        # Add the correct value to the match column
        df_s['match'] = np.where(df_s['Product name in Amazon'] == df_s['shuffled_Product name in Amazon'], 1, 0)

        # Create internal name column
        df_s['Product name in Amazon'] = np.where(df_s['shuffled_Product name in Amazon']!='', 
                                        df_s['shuffled_Product name in Amazon'],
                                        df_s['Product name in Amazon'])

        df_output = df_output.append(df_s)
        df_output = df_output.drop(columns=['shuffled_Product name in Amazon'])

        i += 1

    return df_output


# In[23]:


#Using the function to create synthetic data from the matched data
df_output = create_synthetic_data(product_matched, 7)
df_output.tail()


# In[26]:


df_output.head()


# In[27]:


#Analysing the count of values based on the categories
df_output.match.value_counts()


# In[28]:


#Exporting data into a csv file for analysis
df_output.to_csv('product_matching_synthetic.csv', index=False)


# In[29]:


product_matched.to_csv('actual_matched_product.csv',index=False)

