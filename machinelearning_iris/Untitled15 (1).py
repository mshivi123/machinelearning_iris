
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[3]:


mydata=datasets.load_iris()


# In[4]:


mydata.keys()


# In[5]:


mydata.data.shape


# In[7]:


mydata.feature_names


# In[8]:


mydata.target


# In[9]:


mydata.target_names


# In[10]:


x_feature=mydata.data


# In[16]:


y_target=mydata.target


# In[17]:


x_train=x_feature[:-10]


# In[26]:


x_test=x_feature[-10:]


# In[27]:


y_train=y_target[:-10]


# In[28]:


y_test=y_target[-10:]


# In[29]:


from sklearn.neighbors import KNeighborsClassifier


# In[30]:


myobj=KNeighborsClassifier()


# In[31]:


mymodel=myobj.fit(x_train,y_train)


# In[32]:


ya=y_test


# In[33]:


yp=mymodel.predict(x_test)


# In[34]:


ya-yp


# In[35]:


ya


# In[36]:


yp


# In[38]:


mymodel.predict([[6.4,1.1,5.2,6.2]])


# In[39]:


mydata.target_names[2]


# In[40]:


from sklearn import metrics


# In[41]:


metrics.accuracy_score(ya,yp)*100

