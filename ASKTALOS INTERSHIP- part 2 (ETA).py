#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the file using pandas


# In[2]:


import pandas as pd 


# In[3]:


dataset = pd.read_csv('Material Compressive Strength Experimental Data.csv')


# In[4]:


dataset


# In[5]:


#dimesdimesnsion of the data


# In[6]:


dataset.shape


# In[7]:


#data description


# In[8]:


dataset.describe()


# In[9]:


#dataset info


# In[10]:


dataset.info()


# In[11]:


#finding variant columns


# In[12]:


dataset.var()


# In[13]:


dataset.var().sort_values(ascending = False)


# In[14]:


#finding correlated columns


# In[15]:


corr_matrix = dataset.corr()
corr_matrix 


# In[16]:


# finding columns that carry more than 50% of the same information by setting threshold


# In[17]:


threshold = 0.5
correlated_columns = set()


# In[18]:


for row in range(len(corr_matrix)):
    for col in range(row):
        if abs (corr_matrix.iloc[row][col]) > threshold:
            
            corr = correlated_columns.add(corr_matrix.columns[row])
            print(f'correlated column',corr)
        else:
            print('There are no correlated columns')


# OBSERVATION :
# 
# Thus, there are no columns that are related columns. All columns are mutually exclusive of each other.

# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt 
sns.heatmap(dataset.corr(),annot = True)
plt.show()


# OBSERVATION:
#     
# The above heatmap clearly tells that most of the columns are very less likely to be related to each other by 30%

# In[20]:


#null values count


# In[21]:


dataset.isnull().sum()


# In[22]:


#109 null values ate present in 8 columns which is significant amount of data to be dropped. So,filling with mean would be best here.


# In[23]:


#filling Material Quantity (gm)


# In[24]:


dataset['Material Quantity (gm)'].mean()


# In[25]:


dataset['Material Quantity (gm)'] = dataset['Material Quantity (gm)'].fillna(dataset['Material Quantity (gm)'].mean())


# In[26]:


dataset['Material Quantity (gm)'].isnull().sum()


# OBSERVATION :
# 
# The null values in 'Material Quantity (gm)' is filled with its mean value.

# In[27]:


#filling Additive Catalyst (gm)


# In[28]:


add_mean = dataset['Additive Catalyst (gm)'].mean()


# In[29]:


dataset['Additive Catalyst (gm)'] = dataset['Additive Catalyst (gm)'].fillna(add_mean)


# In[30]:


dataset['Additive Catalyst (gm)'].isnull().sum()


# OBSERVATION :
# 
# The null values in 'Additive Catalyst (gm)' is filled with its mean value.

# In[31]:


#filling Ash Component (gm)


# In[32]:


ash_mean = dataset['Ash Component (gm)'].mean()
dataset['Ash Component (gm)'] = dataset['Ash Component (gm)'].fillna(ash_mean)
dataset['Ash Component (gm)'].isnull().sum()


# OBSERVATION :
# The null values in 'Ash Component (gm)' is filled with its mean value.

# In[33]:


#filling Water Mix (ml)


# In[34]:


wat_mean = dataset['Water Mix (ml)'].mean()
dataset['Water Mix (ml)'] = dataset['Water Mix (ml)'].fillna(wat_mean)
dataset['Water Mix (ml)'].isnull().sum()


# OBSERVATION :
# 
# The null values in 'Water Mix (ml)' is filled with its mean value.

# In[35]:


#filling Plasticizer (gm)


# In[36]:


pla_mean = dataset['Plasticizer (gm)'].mean()
dataset['Plasticizer (gm)'] = dataset['Plasticizer (gm)'].fillna(pla_mean)
dataset['Plasticizer (gm)'].isnull().sum()


# OBSERVATION : The null values in 'Plasticizer (gm)' is filled with its mean value.

# In[37]:


#filling Moderate Aggregator


# In[38]:


mod_mean = dataset['Moderate Aggregator'].mean()
dataset['Moderate Aggregator'] = dataset['Moderate Aggregator'].fillna(mod_mean)
dataset['Moderate Aggregator'].isnull().sum()


# OBSERVATION: 
# 
# The null values in 'Moderate Aggregator' is filled with its mean value.

# In[39]:


#filling Refined Aggregator  


# In[40]:


ref_mean = dataset['Refined Aggregator'].mean()
dataset['Refined Aggregator'] = dataset['Refined Aggregator'].fillna(ref_mean)
dataset['Refined Aggregator'].isnull().sum()


# OBSERVATION : The null values in 'Refined Aggregator' is filled with its mean value.

# In[41]:


#filling Formulation Duration (hrs) 


# In[42]:


for_mean = dataset['Formulation Duration (hrs)'].mean()
dataset['Formulation Duration (hrs)'] = dataset['Formulation Duration (hrs)'].fillna(for_mean)
dataset['Formulation Duration (hrs)'].isnull().sum()


# OBSERVATION:
# 
# The null values in 'Refined Aggregator' is filled with its mean value.

# In[43]:


dataset.isnull().sum()


# OBSERVATION: 
# 
# All the null values are filled.

# In[44]:


#checking skeweness of data


# In[45]:


dataset.skew()


# NOTES 
# -0.5 and 0.5, the distribution of the value is almost symmetrical.
# -1 and -0.5, the data is negatively skewed.
# 0.5 to 1, the data is positively skewed.

# OBSERVATION 
# 
# Almost symmetrical - Compression Strength MPa 
# Negatively skewed -  Compression Strength MPa
# 
# we can say Compression Strength MPa is almost symmetrical but negatively screwed

# In[46]:


import seaborn as sns


# In[47]:


sns.pairplot(dataset)


# OBSERVATION:
#     
# The above graph explains relationship between two variables

# In[48]:


import matplotlib.pyplot as plt
dataset.plot(kind = 'density')
plt.title('Density graph')
plt.xlabel('Dataset')
plt.show()


# OBSERVATION:
#     
# Data is distributed highly in Plasticizer(gm) and Compression Strength Mpa.
# Data is distributed less in Addictive Catalyst (gm) and Material Qunatity (gm)

# In[ ]:


#distance plot


# In[112]:


sns.displot(dataset, legend=True)
plt.show()


# OBSERVATION:
# 
# The data distribution of all the variables against the density distribution is observed.

# In[ ]:


#relationship between all the Columns(IV) Vs Compression Strength MPa (DV)


# In[120]:


for col in dataset:
    sns.lmplot(x=col, y= 'Compression Strength MPa', data=dataset, order=1)
    plt.ylabel("Compression Strength MPa")
    plt.xlabel(col)


# OBSERVATION:
#     
# The IV(s) and DV shows non linear relationship with each other except itself.

# In[50]:


#Detecting outliers


# In[51]:


def find_outliers_IQR(dataset):

   q1= dataset.quantile(0.25)

   q3= dataset.quantile(0.75)

   IQR=q3-q1

   outliers = dataset[((dataset<(q1-1.5*IQR)) | (dataset>(q3+1.5*IQR)))]

   return outliers


outliers = find_outliers_IQR(dataset)

print("number of outliers: "+ str(len(outliers)))

print("max outlier value: "+ str(outliers.max()))

print("min outlier value: "+ str(outliers.min()))

outliers


outliers = find_outliers_IQR(dataset)

outliers


# OBSERVATION:
# 
# number of outliers: 6139
# 
# Dropping the outliers will completely diminish the data.

# In[52]:


#checking number of outliers in each column


# In[53]:


for col in dataset:
    outliers = find_outliers_IQR(dataset[col])
    print(outliers.dtype)
    print(f"the total number of outliers present in {col} is", len(outliers))
  


# OBSERVATION:
# 
# Compression Strength MPa is the only column with 64 outliers

# In[54]:


#specifically observing the outliers in the Compression Strength MPa columnn


# In[55]:


outliers = find_outliers_IQR(dataset['Compression Strength MPa'])
outliers


# In[56]:


#detecting outliers using scaling


# In[57]:


#Trial 1: Using robust scaler which is more effective in case of outliers


# In[58]:


from sklearn.preprocessing import RobustScaler


# In[59]:


transformer = RobustScaler().fit(dataset)
transformer


# In[60]:


RobustScaler()
ro_scaler = transformer.transform(dataset)
ro_scaler


# In[61]:


for col in dataset:
    outliers = find_outliers_IQR(dataset[col])
    print(outliers.dtype)
    print(f"the total number of outliers present in {col} is", len(outliers))


# In[62]:


#Method: Using histogram


# In[63]:


plt.subplot(2,1,1)
plt.hist(dataset)
plt.show()

plt.subplot(2,1,2)
plt.hist(ro_scaler)
plt.show()


# OBSERVATION:
#     
# Although the robust scaler did not reduce the outlier, it has normalised the dataset (Gausssian distribution)

# In[64]:


#Method 2: Box plot


# In[65]:


import plotly.express as px


# In[66]:


px.box(ro_scaler)


# OBSERVATION:
#     
# The ouliers are most significant in Compression Strength Mpa

# In[67]:


px.scatter(ro_scaler)


# OBSERVATION:
# 
# Most of the outliers are scattered in the range -1 to -2.5 which is of Variable 8, that is Compression Strength Mpa

# OBSERVATION:
#     
# The data is scaled between the range -1 to 1

# In[68]:


#resolving outliers


# In[69]:


#Capping


# In[70]:


upper_limit = dataset['Compression Strength MPa'].mean() + 3*dataset['Compression Strength MPa'].std()

print(upper_limit)

lower_limit = dataset['Compression Strength MPa'].mean() - dataset['Compression Strength MPa'].std()

print(lower_limit)


# In[71]:


#before capping
dataset['Compression Strength MPa'].describe()


# In[72]:


#applying the limits


# In[73]:


import numpy as np


# In[74]:


dataset['Compression Strength MPa'] = np.where(dataset['Compression Strength MPa'] > upper_limit,
                                               upper_limit,

       np.where(dataset['Compression Strength MPa'] < lower_limit,  lower_limit,

     dataset['Compression Strength MPa']))


# In[75]:


dataset['Compression Strength MPa'].describe()


# OBSERVATION:
# 
# Limits are applied.

# In[76]:


#inspecting outliers

outliers = find_outliers_IQR(dataset['Compression Strength MPa'])
outliers


# OBSERVATION:
# 
# The limits are applied on the outliers in Compression Strength MPa making it a part of the data and not outliers anymore.

# In[77]:


for col in dataset:
    outliers = find_outliers_IQR(dataset[col])
    print(outliers.dtype)
    print(f"the total number of outliers present in {col} is", len(outliers))
  


# OBSERVATION:
# 
# Thus, the outliers has not affected other columns but Compression Strength MPa. There are no outliers in Compression Strength MPa

# In[78]:


#scaling the converted outlier 


# In[79]:


transformer = RobustScaler().fit_transform(dataset)
transformer


# In[80]:


scaled_dataset = pd.DataFrame(transformer, columns = dataset.columns)


# In[81]:


scaled_dataset

OBSERVATION

All the columns have been scaled accordingly between -1 to 1.
# In[82]:


#Plotting scaled data


# In[83]:


plt.subplot(2,1,1)
plt.hist(ro_scaler)
plt.title("with outliers")
plt.show()

plt.subplot(2,1,2)
plt.hist(scaled_dataset)
plt.title("without outliers")
plt.show()


# OBSERVATION:
#     
# All the outliers present has been resolved.

# In[84]:


import plotly.express as px
df = px.data.tips()
fig1 = px.box(ro_scaler, title = "with outliers")
fig2 = px.box(scaled_dataset, title = "without outliers")
fig1.show()
fig2.show()


# OBSERVATION:
# 
# All the outliers present has been resolved.

# In[85]:


import plotly.express as px
df = px.data.tips()
fig1 = px.scatter(ro_scaler, title = "with outliers")
fig2 = px.scatter(scaled_dataset, title = "without outliers")
fig1.show()
fig2.show()


# OBSERVATION:
# 
# All the outliers present has been resolved.

# In[86]:


#QQ Plot

#NEED: To check if all columns of the data is normally distributed.


# In[87]:


import statsmodels.api as sm
for col in scaled_dataset:
    print(col)
    sm.qqplot(scaled_dataset[col], line = '45')
    plt.show()


# OBSERVATION:
#     
# All the columns are partially normally distributed.

# In[88]:


#converting the columns as normally distributed as possible


# In[89]:


from scipy.stats import norm
import statistics


# In[90]:


scaled_dataset.describe()


# In[91]:


for col in dataset:
    print(col)
    mean = statistics.mean(scaled_dataset[col])
    sd = statistics.stdev(scaled_dataset[col])
    plt.plot(scaled_dataset[col], norm.pdf(scaled_dataset[col], mean, sd))
    plt.show()


# OBSERVATION:
#     
# All the columns are now normally distributed with some skewness.

# In[92]:


dataset.skew()


# OBSERVATION:
#     
# The skewness of the data has not increased and remains the same which is good.

# In[94]:


#power transformation 


# In[95]:


from sklearn.preprocessing import PowerTransformer


# In[124]:


pt = PowerTransformer(method= 'yeo-johnson', standardize=True) 

skl_yeojohnson = pt.fit(dataset)

skl_yeojohnson = pt.transform(dataset)

skl_yeojohnson


# In[125]:


#transforming into dataframe

power_dataset = pd.DataFrame(data=skl_yeojohnson, columns=dataset.columns)

power_dataset


# In[97]:


import statsmodels.api as sm
for col in power_dataset:
    print(col)
    sm.qqplot(power_dataset[col], line = '45')
    plt.show()


# OBSERVATION:
#     
# The columns are more gaussian distributed than before

# In[ ]:




