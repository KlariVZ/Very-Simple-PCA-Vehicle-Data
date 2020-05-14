#!/usr/bin/env python
# coding: utf-8

# In[1]:


### LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

### ENABLE PLOTTING OF GRAPHS IN JUPYTER
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


### READ DATA SET: PD.READ_CSV
dataset = pd.read_csv('vehicle.csv')


# In[3]:


dataset.head()


# In[4]:


### SHAPE OF THE DATA SET
dataset.shape


# In[5]:


dataset.describe().transpose()


# In[6]:


### TYPES OF DATA
dataset.dtypes


# In[7]:


dataset['class'].value_counts()


# In[8]:


dataset.groupby('class').size()


# In[9]:


### BOXPLOTS
dataset.plot(kind='box', figsize=(20,10))
plt.show()


# In[10]:


dataset.hist(figsize=(15,15))
plt.show()


# In[11]:


dataset.isnull().sum()


# In[12]:


dataset.info()


# In[13]:


for i in dataset.columns[:-1]:
    median_value = dataset[i].median()
    dataset[i] = dataset[i].fillna(median_value)


# In[14]:


dataset.info()


# In[15]:


for col_name in dataset.columns[:-1]:
    q1 = dataset[col_name].quantile(0.25)
    q3 = dataset[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr
    
    dataset.loc[ (dataset[col_name] < low) | (dataset[col_name] > high), col_name] = dataset[col_name].median()


# In[16]:


dataset.plot(kind='box', figsize=(20,10))


# In[17]:


### PAIRPLOT

sns.pairplot(dataset,diag_kind='kde')


# In[34]:


### CORRELATION
dataset.corr()


# In[19]:


### SCALE THE DATA WITH STANDARD SCALER
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_df = scaler.fit_transform(dataset.drop(columns = 'class'))


# In[20]:


### CREATE TRAIN AND TEST DATA SETS WITH SCALED DATA
X = scaled_df
y = dataset['class']

X_train, X_test, Y_train, Y_test = train_test_split(X,y, random_state = 10)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[21]:


model = SVC(gamma = 'auto')

### FIT MODEL TO TRAIN DATA
model.fit(X_train,Y_train)

score_using_actual_attributes = model.score(X_test, Y_test)

print(score_using_actual_attributes)


# In[22]:


model = SVC()

params = {'C': [0.01, 0.1, 0.5, 1], 'kernel': ['linear', 'rbf'], 'gamma' : ['auto', 'scale' ]}

model1 = GridSearchCV(model, param_grid=params, verbose=5)

model1.fit(X_train, Y_train)

print("Best Hyper Parameters:\n", model1.best_params_)


# In[23]:


model = SVC(C=1, kernel="rbf", gamma='auto')

scores = cross_val_score(model, X, y, cv=10)

CV_score = scores.mean()
print(CV_score)


# In[24]:


#### CREATE PRINCIPLE COMPONENTS ####


# In[25]:


### USE ATTRIBUTES

from sklearn.decomposition import PCA

pca = PCA().fit(scaled_df)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
print(np.cumsum(pca.explained_variance_ratio_))


# In[26]:


### 95% OF VARIANCE: 8 PC'S

pca = PCA(n_components=8)

X = pca.fit_transform(scaled_df)
Y = dataset['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[27]:


### TRAIN SVC ON PC'S

model = SVC(gamma= 'auto')

model.fit(X_train,Y_train)

score_PCs = model.score(X_test, Y_test)

print(score_PCs)


# In[28]:


model = SVC(C=1, kernel="rbf", gamma='auto')

scores = cross_val_score(model, X, y, cv=10)

CV_score_pca = scores.mean()
print(CV_score_pca)


# In[29]:


#### FINAL RESULTS ####


# In[32]:


final_result = pd.DataFrame({'SVC' : ['All scaled attributes', '8 Principle components'],
                      'Accuracy' : [score_using_actual_attributes,score_PCs],
                      'Cross-validation score' : [CV_score,CV_score_pca]})


# In[33]:


final_result


# In[ ]:




