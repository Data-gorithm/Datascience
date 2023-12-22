#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing the file using pandas


# In[6]:


import pandas as pd 


# In[7]:


dataset = pd.read_csv('Material Compressive Strength Experimental Data.csv')


# In[8]:


dataset


# In[9]:


#dimesdimesnsion of the data


# In[10]:


dataset.shape


# In[11]:


#data description


# In[12]:


dataset.describe()


# In[13]:


#dataset info


# In[14]:


dataset.info()


# In[15]:


#finding variant columns


# In[16]:


dataset.var()


# In[17]:


dataset.var().sort_values(ascending = False)


# In[100]:


#finding correlated columns


# In[19]:


corr_matrix = dataset.corr()
corr_matrix 


# In[101]:


# finding columns that carry more than 50% of the same information by setting threshold


# In[102]:


threshold = 0.5
correlated_columns = set()


# In[104]:


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

# In[105]:


#null values count


# In[106]:


dataset.isnull().sum()


# In[107]:


#109 null values ate present in 8 columns which is significant amount of data to be dropped. So,filling with mean would be best here.


# In[108]:


#filling Material Quantity (gm)


# In[109]:


dataset['Material Quantity (gm)'].mean()


# In[110]:


dataset['Material Quantity (gm)'] = dataset['Material Quantity (gm)'].fillna(dataset['Material Quantity (gm)'].mean())


# In[111]:


dataset['Material Quantity (gm)'].isnull().sum()


# OBSERVATION :
# 
# The null values in 'Material Quantity (gm)' is filled with its mean value.

# In[115]:


#filling Additive Catalyst (gm)


# In[116]:


add_mean = dataset['Additive Catalyst (gm)'].mean()


# In[117]:


dataset['Additive Catalyst (gm)'] = dataset['Additive Catalyst (gm)'].fillna(add_mean)


# In[118]:


dataset['Additive Catalyst (gm)'].isnull().sum()


# OBSERVATION :
# 
# The null values in 'Additive Catalyst (gm)' is filled with its mean value.

# In[119]:


#filling Ash Component (gm)


# In[120]:


ash_mean = dataset['Ash Component (gm)'].mean()
dataset['Ash Component (gm)'] = dataset['Ash Component (gm)'].fillna(ash_mean)
dataset['Ash Component (gm)'].isnull().sum()


# OBSERVATION :
# The null values in 'Ash Component (gm)' is filled with its mean value.

# In[121]:


#filling Water Mix (ml)


# In[80]:


wat_mean = dataset['Water Mix (ml)'].mean()
dataset['Water Mix (ml)'] = dataset['Water Mix (ml)'].fillna(wat_mean)
dataset['Water Mix (ml)'].isnull().sum()


# OBSERVATION :
# 
# The null values in 'Water Mix (ml)' is filled with its mean value.

# In[123]:


#filling Plasticizer (gm)


# In[124]:


pla_mean = dataset['Plasticizer (gm)'].mean()
dataset['Plasticizer (gm)'] = dataset['Plasticizer (gm)'].fillna(pla_mean)
dataset['Plasticizer (gm)'].isnull().sum()


# OBSERVATION : The null values in 'Plasticizer (gm)' is filled with its mean value.

# In[125]:


#filling Moderate Aggregator


# In[84]:


mod_mean = dataset['Moderate Aggregator'].mean()
dataset['Moderate Aggregator'] = dataset['Moderate Aggregator'].fillna(mod_mean)
dataset['Moderate Aggregator'].isnull().sum()


# OBSERVATION: 
# 
# The null values in 'Moderate Aggregator' is filled with its mean value.

# In[127]:


#filling Refined Aggregator  


# In[128]:


ref_mean = dataset['Refined Aggregator'].mean()
dataset['Refined Aggregator'] = dataset['Refined Aggregator'].fillna(ref_mean)
dataset['Refined Aggregator'].isnull().sum()


# OBSERVATION : The null values in 'Refined Aggregator' is filled with its mean value.

# In[129]:


#filling Formulation Duration (hrs) 


# In[130]:


for_mean = dataset['Formulation Duration (hrs)'].mean()
dataset['Formulation Duration (hrs)'] = dataset['Formulation Duration (hrs)'].fillna(for_mean)
dataset['Formulation Duration (hrs)'].isnull().sum()


# OBSERVATION:
# 
# The null values in 'Refined Aggregator' is filled with its mean value.

# In[131]:


dataset.isnull().sum()


# OBSERVATION: 
# 
# All the null values are filled.

# In[132]:


#checking skeweness of data


# In[133]:


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

# In[134]:


import seaborn as sns


# In[135]:


sns.pairplot(dataset)


# OBSERVATION:
#     
# The above graph explains relationship between two variables

# In[52]:


import matplotlib.pyplot as plt
dataset.plot(kind = 'density')
plt.title('Density graph')
plt.xlabel('Dataset')
plt.show()


# OBSERVATION:
#     
# Data is distributed highly in Plasticizer(gm) and Compression Strength Mpa.
# Data is distributed less in Addictive Catalyst (gm) and Material Qunatity (gm)

# In[94]:


sns.displot(dataset, legend=True)


# In[137]:


#scaling data
#NEED FOR SCALING :Making dataset normally distributed for further ML modelling.


# In[138]:


from sklearn.preprocessing import StandardScaler 


# In[139]:


scaler = StandardScaler()


# In[140]:


scaled_data = scaler.fit_transform(dataset)


# In[141]:


scaled_data


# OBSERVATION:
#     
# The data is scaled between the range -1 to 1

# In[ ]:




