#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of banking variables for a Private Financial firm.

# In[1]:


import numpy as np                 # for the 2-d matrices
import pandas as pd                # for the data frames
import matplotlib.pyplot as plt    #for data vizualizations 
import datetime as dt              # for the datetime data
import seaborn as sns


# In[2]:


data = pd.read_csv("LoansData.csv")


# In[3]:


data


# In[6]:


data.head()


# In[8]:


data.shape


# # EDA ANALYSIS - EXPLORATORY DATA ANALYSIS

# ### Step 1: Naming convention

# In[9]:


data.columns


# In[10]:


data.info()


# In[11]:


data.iloc[1:4,]


# In[12]:


data.head(2)


# In[13]:


data.columns = [i.replace('.','_') for i in data.columns]


# In[14]:


data


# ### Step 2: Check the information: Data type Conversions

# In[15]:


data.head()


# In[16]:


data['Interest_Rate'] = data['Interest_Rate'].str.replace('%','').astype('float')


# In[17]:


data


# In[18]:


data.info()


# In[19]:


data['Employment_Length']


# In[20]:


data['FICO_Range'].str.split('-',expand = True)


# In[21]:


data['FICO_Range'] = data['FICO_Range'].str.split('-',expand = True)[0].astype('float')


# In[22]:


data.head()


# In[23]:


data.info()


# # Data type conversions:

# In[25]:


data.head(5)


# In[26]:


data['Loan_Length'] = data['Loan_Length'].str.replace('months','').astype('float')
data['Debt_To_Income_Ratio'] = data['Debt_To_Income_Ratio'].str.replace('%','').astype('float')


# In[27]:


data.head()


# In[28]:


data['Employment_Length'] = data['Employment_Length'].str.replace('years','').str.replace('year','').str.replace('<','').str.replace('+','').astype('float')


# In[29]:


data.head()


# # Checking the relevant and irrelevant variables in the first phase of EDA:

# In[30]:


data.info()


# In[31]:


data.nunique() 


# In[32]:


data['Loan_Length'].value_counts() 


# # Data Duplicacy: Chech if their is duplicate data or not: DDT ( Data Duplicacy Treatment)

# In[33]:


data.duplicated().value_counts()  


# In[34]:


data.duplicated().sum()


# ### How to rename the variables?

# In[35]:


data.columns


# In[36]:


data = data.rename(columns = {'LoanID':'Loan_id','State':'States'})


# In[37]:


data


# In[38]:


data.columns


# # Missing Values Treatment:

# In[39]:


data.Amount_Requested.isna().sum() 


# In[40]:


data.shape


# In[41]:


data.isna().sum()


# In[42]:


(data.isna().sum()/data.shape[0])*100


# In[43]:


def miss_value_treat(s):
    if s.dtype == 'O':
        s = s.fillna(s.mode())
    else:
        s = s.fillna(s.median())
    return s


# In[44]:


data = data.apply(miss_value_treat)


# In[45]:


data.isna().sum()


# In[46]:


data['Amount_Requested'] = data.Amount_Requested.fillna(data['Amount_Requested'].median())


# In[47]:


data['Amount_Funded_By_Investors'] = data.Amount_Funded_By_Investors.fillna(data['Amount_Funded_By_Investors'].median())


# In[48]:


data.isna().sum()


# In[49]:


data.info()


# # Separating the categorical Variables and Numerical variables into two different datasets for Data Preparations for Data Analysis.

# In[51]:


categorical = [var for var in data.columns if data[var].dtype == 'O']


# In[52]:


categorical


# In[53]:


cat_data = data[categorical]


# In[54]:


cat_data['Loan_id'] = data['Loan_id']


# In[55]:


cat_data


# In[56]:


cat_data.shape


# In[57]:


numerical = [var for var in data.columns if data[var].dtype != 'O']


# In[58]:


num_data = data[numerical]


# In[59]:


num_data.head()


# # Filling missing values in Numerical data with the Median values.

# In[60]:


for i in num_data.columns:
        num_data[i] = num_data[i].fillna(num_data[i].median())


# In[61]:


num_data.isna().sum()


# # Filling missing values in Categorical data with the Mode values.

# In[62]:


for i in cat_data.columns:
        cat_data[i] = cat_data[i].fillna(cat_data[i].mode())


# In[63]:


cat_data.isna().sum()


# In[64]:


cat_data.Home_Ownership.count()


# In[65]:


cat_data.Home_Ownership.value_counts()


# In[66]:


cat_data.loc[cat_data['Home_Ownership'] == None]


# In[67]:


s = pd.Series(cat_data.Home_Ownership.isna())
s[s == True]


# In[68]:


cat_data.loc[2492,]


# In[69]:


cat_data['Home_Ownership']  = cat_data['Home_Ownership'].replace("NaN",None)


# In[70]:


cat_data.Home_Ownership.value_counts()


# In[71]:


cat_data.isna().sum()


# In[72]:


cat_data.tail(10)


# In[73]:


cat_data = cat_data.drop(2492)


# In[74]:


cat_data.tail(10)


# In[75]:


cat_data.isna().sum()


# In[76]:


cat_data.shape


# In[77]:


num_data = num_data.drop(2492)


# In[78]:


num_data.shape


# # Outlier Treatment

# In[79]:


num_data.head()


# ### Just observing a single column for the outliers:

# In[80]:


num_data.describe()


# In[81]:


min_Amt = num_data.Amount_Requested.min()
min_Amt


# In[82]:


max_Amt = num_data.Amount_Requested.max()
max_Amt


# ### **quantile() helps to find the percentile vaues at each percentile mark.**

# In[83]:


median = num_data.Amount_Requested.quantile(0.5)
median


# In[84]:


f_q = num_data.Amount_Requested.quantile(0.25)
t_q = num_data.Amount_Requested.quantile(0.75)
p_1 = num_data.Amount_Requested.quantile(0.01)
p_99 = num_data.Amount_Requested.quantile(0.99)


# In[85]:


print("First quartile:",f_q)
print("Third quartile:",t_q)
print("Bottom 1%ile cutoff:",p_1)
print("Top 1%ile cutoff:",p_99)


# In[86]:


iqr = 17000 - 6000


# In[87]:


iqr


# In[88]:


lc = 6000 - (1.5*iqr)
lc


# In[89]:


uc = 17000 + (1.5*iqr)
uc


# In[90]:


sns.boxplot(num_data.Amount_Requested)


# In[91]:


num_data.Loan_Length.nunique()


# In[92]:


num_data.Loan_Length.value_counts()


# In[93]:


num_data.columns


# In[94]:


num_data.Inquiries_in_the_Last_6_Months.value_counts()


# In[95]:


for col in num_data:
    sns.boxplot(num_data[col])
    plt.show()


# In[96]:


num_data2  = num_data.copy()


# In[97]:


num_data2


# In[98]:


lc


# In[99]:


uc


# In[100]:


num_data2['Amount_Requested'] = num_data2.Amount_Requested.clip(lower = lc, upper = uc)


# In[101]:


sns.boxplot(num_data.Amount_Requested)


# In[102]:


sns.boxplot(num_data2.Amount_Requested)


# In[103]:


num_data.columns


# In[104]:


sns.boxplot(num_data2.Amount_Funded_By_Investors)


# In[105]:


num_data2['Amount_Requested'] = num_data2.Amount_Funded_By_Investors.clip(lower = lc, upper = uc)


# # The variables which we have to put the outliers

# # Outlier treatment IQR Method

# In[106]:


def outlier_IQR(s):
    m = s.quantile(0.5)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    q_1p = s.quantile(0.01)
    q_99p = s.quantile(0.99)
    iqr = q3 - q1
    lc = q1 - 1.5*iqr
    uc = q3 + 1.5*iqr
    result = pd.Series([m,q1,q3,q_1p,q_99p,iqr,lc,uc])
    result.index = ['median','first_quartile','third_quartile','pc_1','pc_99','iqr','lower_cutoff','upper_cutoff']
    return result


# In[107]:


def outliertreat_IQR(d):
    m = d.quantile(0.5)
    q1 = d.quantile(0.25)
    q3 = d.quantile(0.75)
    q_1p = d.quantile(0.01)
    q_99p = d.quantile(0.99)
    iqr = q3 - q1
    lc = q1 - 1.5*iqr
    uc = q3 + 1.5*iqr
    return lc,uc


# In[108]:


num_data.apply(outliertreat_IQR)


# In[109]:


num_data2.apply(outlier_IQR)


# In[110]:


cutoffs = pd.DataFrame(num_data2.apply(outliertreat_IQR))
cutoffs


# ### **Made two copies of numerical data and will make two models on each of them and will decide later**
# ### **which one is giving better predictiona and minimum RMSPE OR MAPE error.**

# In[111]:


num_data_2a = num_data.copy()
num_data_2b = num_data.copy()


# # Outlier treatment of only required columns:

# In[112]:


num_data_2a.columns


# In[113]:


for col in num_data2:
    sns.boxplot(num_data2[col])
    plt.show()


# In[114]:


num_data2.Inquiries_in_the_Last_6_Months.value_counts()


# In[116]:


l = ['Amount_Requested','Amount_Funded_By_Investors','Interest_rate','FICO_Range','Open_CREDIT_Lines','Inquiries_in_the_Last_6_Months']


# In[117]:


num_data.apply(outliertreat_IQR)


# In[118]:


num_data_2a['Amount_Requested'] = num_data_2a.Amount_Requested.clip(lower = -10500.00, upper = 33500)


# In[119]:


sns.boxplot(num_data_2a.Amount_Requested)


# In[ ]:





# In[120]:


num_data_2a['Amount_Funded_By_Investors'] = num_data_2a.Amount_Funded_By_Investors.clip(lower = -9000, upper = 31000)
sns.boxplot(num_data_2a.Amount_Requested)


# In[ ]:





# In[121]:


num_data_2a['Interest Rate'] = num_data_2a.Interest_Rate.clip(lower = 1.70, upper = 24.26)
sns.boxplot(num_data_2a.Interest_Rate)


# In[122]:


num_data_2a['FICO_Range'] = num_data_2a.FICO_Range.clip(lower = 612.5, upper = 792.5)
sns.boxplot(num_data_2a.FICO_Range)


# In[123]:


num_data_2a['Open_CREDIT_Lines'] = num_data_2a.Open_CREDIT_Lines.clip(lower = -2.0, upper = 22.0)
sns.boxplot(num_data_2a.Open_CREDIT_Lines)


# In[124]:


num_data_2a['Inquiries_in_the_Last_6_Months'] = num_data_2a.Inquiries_in_the_Last_6_Months.clip(lower = -1.5, upper = 2.5)
sns.boxplot(num_data_2a.Inquiries_in_the_Last_6_Months)


# In[125]:


num_data_2a


# ### Outlier treatment on all columns in num_data_2b

# In[126]:


num_data_2b.Amount_Requested.quantile(0.25)


# In[127]:


for col in num_data_2b.columns:
    q1 = num_data_2b[col].quantile(0.25)
    q3 = num_data_2b[col].quantile(0.75)
    iqr = q3 - q1
    lc = q1 - 1.5*iqr
    uc = q3 + 1.5*iqr
    num_data_2b[col] = num_data_2b[col].clip(lower = lc, upper = uc)


# In[128]:


for col in num_data_2b:
    sns.boxplot(num_data_2b[col])
    plt.show()


# In[129]:


num_data_2a.head(2)


# In[130]:


num_data_2a.shape


# In[131]:


num_data_2a = num_data_2a.drop(columns = 'Interest Rate')


# In[132]:


num_data_2a.shape


# In[133]:


num_data_2b.shape


# In[134]:


num_data_2a.columns


# In[135]:


num_data_2b.columns


# In[136]:


num_data_2b.head(2)


# In[137]:


num_data_2b.shape


# In[138]:


cat_data.head(2)


# # Merging the categorical and the numerical data to get the dataset:  

# In[139]:


M_data1 = pd.merge(num_data_2a,cat_data,how = 'inner',left_on = 'Loan_id',right_on = 'Loan_id')


# In[140]:


M_data1.shape


# In[141]:


M_data1.columns


# In[142]:


M_data2 = pd.merge(num_data_2b,cat_data,how = 'inner',left_on = 'Loan_id',right_on = 'Loan_id')


# In[143]:


M_data2.shape


# In[144]:


M_data2.columns


# In[145]:


from scipy.stats import skew


# In[147]:


print(skew(num_data,axis=0,bias=True))


# In[148]:


from scipy.stats import kurtosis


# In[149]:


print(kurtosis(num_data,axis=0,bias=True))


# In[ ]:





# In[150]:


print(data['Loan_Length'].value_counts())
data['Loan_Length'].value_counts().plot(kind = 'bar',color = 'green')
plt.title("Distribution of customers by their loan lengths.")
plt.xlabel("Loan length")
plt.ylabel("No of customers")


# In[ ]:





# In[151]:


print(data['States'].value_counts())
data['States'].value_counts().plot(kind = 'bar',color = 'yellow',figsize = (20,15))
plt.title("Distribution of customers by their State.")
plt.xlabel("States")
plt.ylabel("No of customers")


# In[152]:


print(data['Loan_Purpose'].value_counts())
data['Loan_Purpose'].value_counts().plot(kind = 'bar',color = 'red',figsize = (15,10))
plt.title("Distribution of customers by their Loan_Purpose.")
plt.xlabel("Loan.Purpose")
plt.ylabel("No of customers")


# In[ ]:





# In[153]:


print(data['Home_Ownership'].value_counts())
data['Home_Ownership'].value_counts().plot(kind = 'bar',color = 'blue',figsize = (8,5))
plt.title("Distribution of customers by their Home_Ownership.")
plt.xlabel("Home_Ownership")
plt.ylabel("No of customers")


# In[154]:


data.head(2)


# In[169]:


import pandas as pd
import matplotlib.pyplot as plt
plt.scatter(data['Loan_id'],data['Loan_Purpose'])
plt.title("Scatter Plot")
plt.xlabel('Loan_id')
plt.ylabel('Loan_Purpose')
plt.colorbar()
plt.show()


# In[173]:


import pandas as pd
import matplotlib.pyplot as plt
plt.scatter(data['Employment_Length'],data['Loan_id'])
plt.title("Scatter Plot")
plt.xlabel('Employment_Length')
plt.ylabel('Loan_id')
plt.colorbar()
plt.show()


# In[ ]:





# In[188]:


import pandas as pd
import matplotlib.pyplot as plt
plt.hist(data['Interest_Rate'],color='red')
plt.title("Histogram")
plt.show()


# In[183]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(x='Amount_Funded_By_Investors',data=data,kde=True,hue='Home_Ownership')
plt.show()


# In[ ]:




