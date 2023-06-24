#!/usr/bin/env python
# coding: utf-8

# In[ ]:


HOUSE PRICE PREDICTION-PROJECT


# In[2]:


import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[3]:


usa = pd.read_csv('USA_Housing.csv')


# In[5]:


usa


# In[6]:


usa.info()


# In[7]:


usa.info()


# In[8]:


usa.columns


# In[9]:


y=usa['Price']
X=usa[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[10]:


sb.histplot(usa['Price'])


# In[11]:


sb.pairplot(usa, x_vars = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population'], y_vars = 'Price', kind = 'reg')
pt.show()


# In[12]:


co_matrix = usa[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']].corr()
sb.heatmap(co_matrix, annot = True, cmap = 'viridis')
pt.title('Correlation Matrix')
pt.show()


# In[13]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)


# In[14]:


usalr = LinearRegression()


# In[15]:


usalr.fit(Xtrain,ytrain)


# In[16]:


usalr.coef_


# In[17]:


pd.DataFrame(usalr.coef_,index=X.columns,columns=['mycoef'])


# In[18]:


pr = usalr.predict(Xtest)


# In[19]:


r2_score(ytest,pr)


# In[20]:


mean_absolute_error(ytest,pr)


# In[21]:


mean_squared_error(ytest,pr)


# In[22]:


i = float(input('Enter Average Area Income: '))
a = float(input('Enter Average Area House Age: '))
r = float(input('Enter Average Area Number of Rooms: '))
br = float(input('Enter Average Area Number of Bedrooms: '))
p = float(input('Enter Area Population: '))
print('The predicted House Price is:' ,float(usalr.predict([[i, a, r, br, p]])[0]))


# In[ ]:





# In[ ]:




