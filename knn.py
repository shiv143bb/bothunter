#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.utils import resample

from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours
from itertools import cycle

import joblib
import pickle

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load the CTU_13 dataset
df_CTU_13 = pd.read_csv('CTU_13.csv', index_col=0)

# Display the dataset
df_CTU_13


# In[6]:


df_CTU_13.isnull().sum()


# In[7]:


df_CTU_13 = df_CTU_13.drop(['StartTime','SrcAddr','Sport','DstAddr','Dport','State'], axis=1)
df_CTU_13


# In[8]:


# Replace all NaN values with the number 9
df_CTU_13 = df_CTU_13.fillna(9)

# Display the dataset
df_CTU_13

# Confirm that there are no missing values remaining
df_CTU_13.isnull().sum()


# In[9]:


# Convert label strings to numeric values for modeling
lst = []
for i in df_CTU_13['Label']:
    if 'Botnet' in i:
        lst.append(1)
    elif 'Normal' in i:
        lst.append(2)
    else:
        lst.append(0)

# Update the DataFrame with numeric labels
df_CTU_13['Label'] = lst

# Display count of each label
df_CTU_13['Label'].value_counts()


# In[10]:


# Convert protocol names to standard IP numeric values
protocol_number = []
for i in df_CTU_13['Proto']:
    if i == 'udp':
        protocol_number.append(17)
    elif i == 'tcp':
        protocol_number.append(6)
    elif i == 'icmp':
        protocol_number.append(1)
    else:
        protocol_number.append(0)

# Update the DataFrame with numeric protocol values
df_CTU_13['Proto'] = protocol_number


# In[11]:


# Convert direction symbols to numeric values
# '<->' = 1, '->' = 2, others = 0
direction_number = []
for i in df_CTU_13['Dir']:
    if i == '  <->':
        direction_number.append(1)
    elif i == '   ->':
        direction_number.append(2)
    else:
        direction_number.append(0)

# Update the DataFrame with numeric direction values
df_CTU_13['Dir'] = direction_number

# Display the updated dataset
df_CTU_13


# In[12]:


A = df_CTU_13.copy()


# In[13]:


X_new_df = A.drop(['Label'], axis=1).copy()
y = A['Label'].copy()


# In[14]:


def draw_plot(col):
    # Convert to string for proper categorical plotting if needed
    if col.name == 'sTos' or col.name == 'dTos':
        col = col.astype('str')
    
    names = col.value_counts().index.tolist()  # Unique values
    values = col.value_counts().tolist()       # Counts
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(names, values, color='Blue')
    ax.set_title(f'{col.name} values')
    
    # Add count labels on top of bars
    for index, data in enumerate(values):
        plt.text(x=index, y=data + 50000, s=f"{data}", fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.show()

one_hot_lst = ['Dir', 'Proto', 'sTos', 'dTos']
for col_name in one_hot_lst:
    draw_plot(A[col_name])


# In[16]:


one_hot_lst = ['Dir','Proto','sTos','dTos']
column_list = ['Dur','TotPkts','TotBytes','SrcBytes',
               'Dir_   ->','Dir_  <->','Dir_others',
	             'Proto_icmp','Proto_tcp','Proto_udp','Proto_others',
               'sTos_0.0','sTos_9.0','sTos_others','dTos_0.0','dTos_9.0','dTos_others','Label']
def one_hot_encoding(df):
    df_new = pd.DataFrame()
    other_list = []
    for i in one_hot_lst:
        if i == 'Dir':
            for j in df[i]:
                if j == '  <->' or j == '   ->':
                    other_list.append(0)  # Known categories
                else:
                    other_list.append(1)  # "Others" category
            other_name = 'Dir_others'
            df[other_name] = other_list
            other_list.clear()
            df_new = pd.get_dummies(df, columns=[i])  # One-hot encode 'Dir'

        elif i == 'Proto':
            for j in df[i]:
                if j in ['tcp', 'udp', 'icmp']:
                    other_list.append(0)
                else:
                    other_list.append(1)
            other_name = 'Proto_others'
            df_new[other_name] = other_list
            other_list.clear()
            df_new = pd.get_dummies(df_new, columns=[i])  # One-hot encode 'Proto'

        elif i == 'sTos':
            for j in df[i]:
                if j == 0 or j == 9:
                    other_list.append(0)
                else:
                    other_list.append(1)
            other_name = 'sTos_others'
            df_new[other_name] = other_list
            other_list.clear()
            df_new = pd.get_dummies(df_new, columns=[i])  # One-hot encode 'sTos'

        elif i == 'dTos':
            for j in df[i]:
                if j == 0 or j == 9:
                    other_list.append(0)
                else:
                    other_list.append(1)
            other_name = 'dTos_others'
            df_new[other_name] = other_list
            other_list.clear()
            df_new = pd.get_dummies(df_new, columns=[i])  # One-hot encode 'dTos'

    # Keep only the columns in column_list, drop all others
    for col in df_new.columns:
        if col not in column_list:
            df_new = df_new.drop([col], axis=1)
    return df_new
A = one_hot_encoding(A)


# In[17]:


X = A.drop(['Label'], axis=1).copy()
y = A['Label'].copy()


# In[18]:


X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
print(X_new.shape)
X_new


# In[19]:


X_new_df = pd.DataFrame(X_new)
X_new_df.describe()


# In[20]:


X_new = SelectKBest(f_classif, k=10).fit_transform(X, y)
print(X_new.shape)
X_new


# In[21]:


X_new_df = pd.DataFrame(X_new)
X_new_df.describe()


# In[22]:


rus = RandomUnderSampler(random_state=0, sampling_strategy={0: 801132, 1: 444699, 2: 356433})
X_res, y_res = rus.fit_resample(X_new_df, y)


# In[23]:


y_res.value_counts()


# In[24]:


X_res


# In[25]:


nearmiss = NearMiss(version=1, sampling_strategy={0: 801132, 1: 444699, 2: 356433})
X_res, y_res = nearmiss.fit_resample(X_new_df, y)


# In[26]:


y_res.value_counts()


# In[27]:


X_res


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size=0.8, test_size=0.2, random_state=0)


# In[29]:


X_test.value_counts()


# In[30]:


y_test.value_counts()


# In[31]:


X_train.value_counts()


# In[32]:


y_train.value_counts()


# In[33]:


# fit scaler on training data
scaler = MinMaxScaler(feature_range=(0,1)).fit(X_train)
# transform training data
X_train = scaler.transform(X_train)
# transform testing dataabs
X_test = scaler.transform(X_test)


# In[34]:


# fit scaler on training data
scaler = StandardScaler().fit(X_train)
# transform training data
X_train = scaler.transform(X_train)
# transform testing dataabs
X_test = scaler.transform(X_test)


# In[35]:


model_DT = DecisionTreeClassifier(random_state=0)
model_DT.fit(X_train, y_train)


# In[36]:


y_predict_DT = model_DT.predict(X_test)
acc_score_DT = accuracy_score(y_test, y_predict_DT)
print("Accuracy score: {} %".format(acc_score_DT*100))
bot_recall_score = recall_score(y_test, y_predict_DT, average=None)[1]
bot_precision_score = precision_score(y_test, y_predict_DT, average=None)[1]
bot_f1_score = f1_score(y_test, y_predict_DT, average=None)[1]
print('Botnet traffic recall score: {}'.format(bot_recall_score))
print('Botnet traffic precision score: {}'.format(bot_precision_score))
print('Botnet traffic f1 score: {}'.format(bot_f1_score))


# In[38]:


y_predict_DT


# In[39]:


joblib.dump(model_DT, 'saved_models/decision_tree_model.joblib')


# In[40]:


model_RF = RandomForestClassifier(random_state=0)
model_RF.fit(X_train, y_train)


# In[42]:


joblib.dump(model_RF, 'saved_models/RandomForestModel.joblib')


# In[43]:


y_predict_RF = model_RF.predict(X_test)
acc_score_RF = accuracy_score(y_test, y_predict_RF)
print("Accuracy score: {} %".format(acc_score_RF*100))
bot_recall_score = recall_score(y_test, y_predict_RF, average=None)[1]
bot_precision_score = precision_score(y_test, y_predict_RF, average=None)[1]
bot_f1_score = f1_score(y_test, y_predict_RF, average=None)[1]
print('Botnet traffic recall score: {}'.format(bot_recall_score))
print('Botnet traffic precision score: {}'.format(bot_precision_score))
print('Botnet traffic f1 score: {}'.format(bot_f1_score))


# In[44]:


model_KNN = KNeighborsClassifier()
model_KNN.fit(X_train, y_train)


# In[45]:


y_predict_KNN = model_KNN.predict(X_test)
acc_score_KNN = accuracy_score(y_test, y_predict_KNN)
print("Accuracy score: {} %".format(acc_score_KNN*100))
bot_recall_score = recall_score(y_test, y_predict_KNN, average=None)[1]
bot_precision_score = precision_score(y_test, y_predict_KNN, average=None)[1]
bot_f1_score = f1_score(y_test, y_predict_KNN, average=None)[1]
print('Botnet traffic recall score: {}'.format(bot_recall_score))
print('Botnet traffic precision score: {}'.format(bot_precision_score))
print('Botnet traffic f1 score: {}'.format(bot_f1_score))


# In[46]:


joblib.dump(model_KNN, 'saved_models/KNNModel.joblib')


# In[ ]:




