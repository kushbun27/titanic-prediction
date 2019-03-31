
# coding: utf-8

# ___
# # Logistic Regression with Python
# [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This is a very famous data set 
# 
# We'll be trying to predict a classification- survival or deceased.
# 
# 
# ## Import Libraries
# 

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## The Data
# 
# 

# In[2]:

train = pd.read_csv('titanic_train.csv')


# In[3]:

train.head()


# # Exploratory Data Analysis
# 
# 
# ## Missing Data
# 
# 

# In[4]:

train[train['Sex']=='female']['Embarked'].value_counts()


# 

# In[5]:

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[6]:

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[7]:

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[8]:

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[9]:

train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[10]:

train.corr()


# In[11]:

sns.heatmap(train.corr())


# In[ ]:




# In[12]:

sns.countplot(x='SibSp',data=train)


# In[13]:

train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# 

# In[ ]:




# In[ ]:




# ___
# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:
# 

# In[14]:

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[15]:

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# Now apply that function!

# In[16]:

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# 

# In[17]:

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# 

# In[18]:

train.drop('Cabin',axis=1,inplace=True)


# In[19]:

train.head()


# In[20]:

train.dropna(inplace=True)


# ## Converting Categorical Features 
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[21]:

train.info()


# In[22]:

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[23]:

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[24]:

train = pd.concat([train,sex,embark],axis=1)


# In[25]:

train.head()


# 
# # Building a Logistic Regression model
# 
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
# 
# ## Train Test Split

# In[26]:

from sklearn.model_selection import train_test_split


# In[27]:

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# ## Training and Predicting

# In[28]:

from sklearn.linear_model import LogisticRegression


# In[29]:

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[30]:

predictions = logmodel.predict(X_test)


# Let's move on to evaluate our model!

# ## Evaluation

# We can check precision,recall,f1-score using classification report!

# In[31]:

from sklearn.metrics import classification_report


# In[32]:

print(classification_report(y_test,predictions))


# 

# In[ ]:



