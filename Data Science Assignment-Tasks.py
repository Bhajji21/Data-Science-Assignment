#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading of dataset
import pandas as pd
import numpy as np


# In[2]:


training_data = pd.read_csv("C:/Users/abhay/OneDrive/Desktop/customer-segmentation-dataset/trainingSet.csv")


# In[3]:


training_data.head()


# In[4]:


testing_data = pd.read_csv("C:/Users/abhay/OneDrive/Desktop/customer-segmentation-dataset/testingSet.csv")


# In[5]:


testing_data.head()


# In[6]:


training_data.values


# In[7]:


#describing dataset
training_data.describe()


# In[8]:


#shape of dataset
training_data.shape


# In[9]:


testing_data.shape


# In[10]:


training_data.groupby('DEFECTIVE').size()


# In[11]:


#label encoding
training_data['DEFECTIVE'].unique()


# In[12]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
training_data['DEFECTIVE']= label_encoder.fit_transform(training_data['DEFECTIVE'])
training_data['DEFECTIVE'].unique()


# In[13]:


#importing library for visualization
import matplotlib.pyplot as plt
training_data.boxplot(column=['NUM_UNIQUE_OPERATORS','NUM_UNIQUE_OPERANDS'])


# In[14]:


#scatter plot
training_data.plot.scatter(x='NUM_UNIQUE_OPERATORS', y='DEFECTIVE', c='blue')


# In[15]:


#scatter plot
training_data.plot.scatter(x='NUM_UNIQUE_OPERANDS', y='DEFECTIVE', c='green')


# In[16]:


#spliting of dataset into training and validation set...... where validation set is equal to 55% of training dataset
import sklearn
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

X = training_data.iloc[:, :-1].values
y = training_data.iloc[:, 13].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.55)


# In[17]:


X_train.shape


# In[18]:


X_val.shape


# In[22]:


##########       PCA      ############
#splitting oroginal dataset
A = training_data.iloc[:, :-1].values
b = training_data.iloc[:, 13].values
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size= 0.2, random_state = 0)

A_train.shape
A_test.shape


# In[23]:


#performing the preprocessing of dataset
from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
A_train = ssc.fit_transform(A_train)
A_test = ssc.fit_transform(A_test)


# In[26]:


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
A_train = pca.fit_transform(A_train)
A_test = pca.transform(A_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)


# In[28]:


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(A_train, b_train)


# In[29]:


#predicting the test set result
y_pred = classifier.predict(A_test)


# In[30]:


#making confusion matrix between test set of b and predicted value. 
from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(b_test, y_pred) 


# In[31]:


print(cm)


# In[33]:


# Predicting the training set result through scatter plot  
from matplotlib.colors import ListedColormap 
  
X_set, y_set = A_train, b_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('yellow', 'white', 'aquamarine'))) 
  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j) 
  
plt.title('Logistic Regression (Training set)') 
plt.xlabel('PC1') # for Xlabel 
plt.ylabel('PC2') # for Ylabel 
plt.legend() # to show legend 
  
# show scatter plot 
plt.show()


# In[34]:


# Predicting the training set 
# result through scatter plot  
from matplotlib.colors import ListedColormap 
  
X_set, y_set = A_test, b_test 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('yellow', 'white', 'aquamarine'))) 
  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j) 
  
plt.title('Logistic Regression (Training set)') 
plt.xlabel('PC1') # for Xlabel 
plt.ylabel('PC2') # for Ylabel 
plt.legend() # to show legend 
  
# show scatter plot 
plt.show()


# In[ ]:




