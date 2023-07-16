#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import itertools
import missingno as msno
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings('ignore')        


# In[7]:


customer=pd.read_csv('C:/Users/HP/AppData/Local/Temp/Rar$DIa12368.11197/Mall_Customers.csv')


# In[5]:


customer.head()


# In[11]:


customer.index


# In[10]:


customer.size


# In[12]:


list(customer.columns)


# In[13]:


customer.shape


# In[14]:


print('number of columns',customer.shape[0])


# In[15]:


print('number of rows',customer.shape[1])


# In[17]:


customer_catg=customer.select_dtypes('object')


# In[18]:


customer_cont=customer.select_dtypes('number')


# In[19]:


customer_catg


# In[38]:


customer.info()


# In[39]:


customer.count()


# In[20]:


customer_cont


# In[22]:


customer_cont.head()


# In[23]:


customer.head(5)


# In[24]:


customer['Gender'].unique()


# In[25]:


customer['Age'].unique()


# In[26]:


customer['Age'].nunique()


# In[28]:


customer['Annual Income (k$)'].unique()


# In[29]:


customer['Annual Income (k$)'].nunique()


# In[32]:


customer['Spending Score (1-100)'].unique()


# In[33]:


customer['Spending Score (1-100)'].nunique()


# ### CLEANING THE DATASET

# In[35]:


customer.duplicated()


# In[37]:


customer.duplicated().any()


# In[41]:


customer.isnull().sum()


# In[42]:


customer.notnull()


# In[79]:


customer.describe()


# ## VISUALIZING MISSING VALUES

# In[43]:


import missingno as msn
msn.matrix(customer)


# In[45]:


msn.bar(customer)


# In[46]:


customer_nullvalue = pd.DataFrame((customer.isnull().sum())*100/customer.shape[0]).reset_index()
customer_nullvalue.columns = ['Column Name', 'Null Values Percentage']
fig = plt.figure(figsize=(18,6))
ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=customer_nullvalue,color='red')
plt.xticks(rotation =90,fontsize =8)
ax.axhline(40, ls='--',color='red')
plt.title("Percentage of Missing values in application data")
plt.ylabel("Null Values PERCENTAGE")
plt.xlabel("COLUMNS")
plt.show()


# In[47]:


sns.heatmap(customer.isnull())


# ## ANALZING THE DATA WITH VISUALS

# ### univariate analysis

# In[49]:


sns.distplot(customer['Age'],kde=False,hist=True,bins=12)
plt.title('Age distribution',size=16)
plt.ylabel('count')


# In[51]:


sns.distplot(customer['Age'])


# In[52]:


customer.head()


# In[55]:


customer.Gender.unique()


# In[63]:


plt.figure(figsize=(8,5))
sns.countplot('Gender', data = customer, saturation=0.9)


# In[65]:


sns.histplot(customer['Annual Income (k$)'],kde=True,bins=15)


# In[66]:


sns.histplot(customer['Spending Score (1-100)'],kde=True,bins=15)


# ### bivariate and multivariate analysis

# In[67]:


customer.head(2)


# In[74]:


sns.stripplot(x='Gender',y='Spending Score (1-100)',data=customer,dodge=True ,palette='YlGnBu')


# In[73]:


sns.boxplot(x='Gender',y='Annual Income (k$)',data=customer,palette='YlGnBu')


# In[75]:


customer.hist()


# In[78]:


plt.figure(figsize=(10,8))
sns.heatmap(customer.corr(), vmin=-1, cmap="plasma_r", annot=True)
#same thing can be seen from the correlation as well


# ## DATA SCIENCE

# ### building the model  using KMEANS CLUSTER ALGORITHMS

# In[81]:


customer.columns


# In[82]:


X=customer[['Annual Income (k$)','Spending Score (1-100)']]


# In[83]:


X


# In[84]:


from sklearn.cluster import KMeans


# In[85]:


k_means=KMeans()


# In[86]:


k_means.fit(X)


# In[87]:


k_means=KMeans()
k_means.fit_predict(X)


# In[88]:


k_means=KMeans(5)


# In[91]:


k_means=KMeans(n_clusters=5)


# In[89]:


k_means=KMeans(2)


# In[92]:


k_means=KMeans(n_clusters=8)


# ####   ELBOW METHOD TO FIND OPTIMAL NUMBER OF CLUSTERS

# In[96]:


wcss=[]
for i in range(1,11):
    k_means=KMeans(n_clusters=i)
    k_means.fit(X)
    wcss.append(k_means.inertia_)


# In[97]:


wcss


# In[100]:


plt.plot(range(1,11),wcss)
plt.title=('Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('WCSS')
plt.show


# ### MODEL TRAINING

# In[101]:


X=customer[['Annual Income (k$)','Spending Score (1-100)']]


# In[102]:


KMeans(n_clusters=5,random_state=42)


# In[105]:


y_means=k_means.fit_predict(X)


# In[106]:


y_means


# In[109]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='red',label='cluster1')
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='red',label='cluster2')
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='red',label='cluster3')
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='red',label='cluster4')
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='red',label='cluster5')
plt.legend()


# In[111]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='red',label='cluster1')
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='yellow',label='cluster2')
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='green',label='cluster3')
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='blue',label='cluster4')
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='black',label='cluster5')
plt.legend()


# In[113]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='red',label='cluster1')
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='yellow',label='cluster2')
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='green',label='cluster3')
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='blue',label='cluster4')
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='black',label='cluster5')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100,c='magenta')
plt.legend()


# In[117]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='red',label='cluster1')
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='yellow',label='cluster2')
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='green',label='cluster3')
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='blue',label='cluster4')
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='black',label='cluster5')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100,c='magenta')
plt.title('CustomerSegmentation')
plt.xlabels('Annual Income')
plt.ylabel('Spending Score')
plt.show()
plt.legend()


# ### perform prediction

# In[118]:


k_means.predict([[15,39]])


# ### SAVE THE MODEL

# In[119]:


import joblib


# In[125]:


joblib.dump(k_means,'customer segmentation')


# In[126]:


model=joblib.load('customer segmentation')


# In[127]:


model


# In[128]:


model.predict([[15,39]])


# ### GUI

# In[1]:


from tkinter import *


# In[2]:


import joblib


# In[3]:



master=Tk()





master.mainloop()


# In[4]:


#ADD TITLE TO IT
master=Tk()

master.title('customer segmentation')



master.mainloop()


# In[5]:


#ADD TITLE TO IT
master=Tk()

master.title('customer segmentation')
master.geometry('400x300')
master.minsize(200,200)
master.maxsize(600,600)
master.mainloop()


master.mainloop()


# In[8]:


#ADD TITLE TO IT
master=Tk()

master.title('customer segmentation')
master.geometry('400x300')
master.minsize(200,200)
master.maxsize(600,600)

master.configure(bg='lightblue')

master.mainloop()


# In[9]:


#ADD TITLE TO IT
master=Tk()

master.title('customer segmentation')
master.geometry('400x300')
master.minsize(200,200)
master.maxsize(600,600)

master.configure(bg='lightblue')
label=Label(master,text='customer segmentation',font=('Arial',20,'bold'),width=10,height=1,bg='lightblue',foreground='yellow').grid(row=0,columnspan=2)

Label(master,text='Annual Income (k$)').grid(row=1)
Label(master,text='Spending Score (1-100)').grid(row=2)

master.mainloop()


# In[10]:


master=Tk()

master.title('customer segmentation')
master.geometry('400x300')
master.minsize(200,200)
master.maxsize(600,600)

master.configure(bg='lightblue')
label=Label(master,text='customer segmentation',font=('Arial',20,'bold'),width=10,height=1,bg='lightblue',foreground='yellow').grid(row=0,columnspan=2)

Label(master,text='Annual Income (k$)').grid(row=1)
Label(master,text='Spending Score (1-100)').grid(row=2)

e1=Entry(master,font=('Arial',14),bg='grey',fg='white',borderwidth=3,show='*')
e2=Entry(master,font=('Arial',14),bg='grey',fg='white',borderwidth=3,show='*')

master.mainloop()


# In[11]:


master=Tk()

master.title('customer segmentation')
master.geometry('400x300')
master.minsize(200,200)
master.maxsize(600,600)

master.configure(bg='lightblue')
label=Label(master,text='customer segmentation',font=('Arial',20,'bold'),width=10,height=1,bg='lightblue',foreground='yellow').grid(row=0,columnspan=2)

Label(master,text='Annual Income (k$)').grid(row=1)
Label(master,text='Spending Score (1-100)').grid(row=2)

e1=Entry(master,font=('Arial',14),bg='grey',fg='white',borderwidth=3,show='*')
e2=Entry(master,font=('Arial',14),bg='grey',fg='white',borderwidth=3,show='*')


e1.grid(row=1,column=1)    
e2.grid(row=2,column=1)

master.mainloop()


# In[13]:


def show_entry():
    p1=float(e1.get())
    p2=float(e2.get())

    model=joblib.load('customer segmentation')
    result=model.predict([[p1,p2]])


    
    Label(master, text='customer segmentation').grid(row=3)
    Label(master,text=result).grid(row=4) 




master=Tk()

master.title('customer segmentation')
master.geometry('400x300')
master.minsize(200,200)
master.maxsize(600,600)

master.configure(bg='lightblue')
label=Label(master,text='customer segmentation',font=('Arial',20,'bold'),width=10,height=1,bg='lightblue',foreground='yellow').grid(row=0,columnspan=2)

Label(master,text='Annual Income (k$)').grid(row=1)
Label(master,text='Spending Score (1-100)').grid(row=2)

e1=Entry(master,font=('Arial',14),bg='grey',fg='white',borderwidth=3)
e2=Entry(master,font=('Arial',14),bg='grey',fg='white',borderwidth=3)


e1.grid(row=1,column=1)    
e2.grid(row=2,column=1)

master.mainloop()


# In[ ]:


def show_entry():
    p1=float(e1.get())
    p2=float(e2.get())

    model=joblib.load('customer segmentation')
    result=model.predict([[p1,p2]])


    
    Label(master, text='customer segmentation').grid(row=3)
    Label(master,text=result).grid(row=4) 




master=Tk()

master.title('customer segmentation')
master.geometry('400x300')
master.minsize(200,200)
master.maxsize(600,600)

master.configure(bg='lightblue')
label=Label(master,text='customer segmentation',font=('Arial',20,'bold'),width=10,height=1,bg='lightblue',foreground='yellow').grid(row=0,columnspan=2)

Label(master,text='Annual Income (k$)').grid(row=1)
Label(master,text='Spending Score (1-100)').grid(row=2)

e1=Entry(master,font=('Arial',14),bg='grey',fg='white',borderwidth=3)
e2=Entry(master,font=('Arial',14),bg='grey',fg='white',borderwidth=3)


e1.grid(row=1,column=1)    
e2.grid(row=2,column=1)

Button(master,text='predict',command=show_entry,bg='pink',font=('Arial',20,'bold'),borderwidth=3,activebackground='blue').grid()

master.mainloop()


# In[ ]:




