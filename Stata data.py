#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sb
import scipy as sc
from sklearn.preprocessing import PowerTransformer
from datetime import datetime, date
import os
import math


# In[170]:


file = 'disclosure_times2020-01-09.csv'
tm_path ='{}\{}\{}'.format(os.getcwd(),'CG_Reports_eng',file)
disclosure_times = pd.read_csv(tm_path, parse_dates=['ref_date','upload_date_jp','upload_date_eng'])
disclosure_times['time_gap'] = disclosure_times['upload_date_eng'] - disclosure_times['upload_date_jp']
disclosure_times['time_gap'] = disclosure_times['time_gap'].map(lambda x: x.days)


# In[171]:


#see time gaps over time in general
disclosure_times['year'] = disclosure_times['ref_date'].map(lambda x: str(x.year))
disclosure_times['year_month'] = disclosure_times['ref_date'].map(lambda x: str(date(x.year,x.month,1)))
# disclosure_times['year_month'] = disclosure_times['ref_date'].round('BQS')
descriptive_stats = pd.concat(
    [disclosure_times[disclosure_times['year']== '2015']['time_gap'].describe(percentiles=[0.5,0.7,0.8,0.9,0.95, 0.99, 0.995]),
     disclosure_times[disclosure_times['year']== '2016']['time_gap'].describe(percentiles=[0.5,0.7,0.8,0.9,0.95, 0.99, 0.995]),
     disclosure_times[disclosure_times['year']== '2017']['time_gap'].describe(percentiles=[0.5,0.7,0.8,0.9,0.95, 0.99, 0.995]),
     disclosure_times[disclosure_times['year']== '2018']['time_gap'].describe(percentiles=[0.5,0.7,0.8,0.9,0.95, 0.99, 0.995]),
     disclosure_times[disclosure_times['year']== '2019']['time_gap'].describe(percentiles=[0.5,0.7,0.8,0.9,0.95, 0.99, 0.995]),
     disclosure_times[disclosure_times['year']== '2020']['time_gap'].describe(percentiles=[0.5,0.7,0.8,0.9,0.95, 0.99, 0.995])],
                                            axis=1)

neg_disclosure_times = disclosure_times[disclosure_times['time_gap'] < 0]
disclosure_times = disclosure_times[disclosure_times['time_gap'] >= 0]
plt.hist(disclosure_times['time_gap'], bins=20,density=True, log=True)
plt.title('Log Transformed histogram of Time Gap Data')
plt.xlabel('Days between English CG Report and Japanese CG Report')
# (array([  3., 926., 264.,  58.,  22.,  13.,   1.,   2.,   1.,   3.]),
#  array([-81. , -36.3,   8.4,  53.1,  97.8, 142.5, 187.2, 231.9, 276.6,
#         321.3, 366. ]),
#  <a list of 10 Patch objects>)
len(disclosure_times)


# In[172]:


neg_disclosure_times


# evalute, they should be able to be left out because the english reports were uplaoded for different reasons from the usual for time lag. perhaps something company and time specific, outliers which can be safely ignored.
# 

# In[161]:


days = 0
print('{}% of english reports have a time gap larger than {} days.'
      .format(round(len(disclosure_times[disclosure_times['time_gap'] > days])/
                    len(disclosure_times['time_gap'])*100),days))
#39% of english reports have a non-zero time gap.


# In[179]:


a = disclosure_times[['year_month','time_gap']].dropna().groupby('year_month').mean()
ax = a[1:-1].rolling(6).mean().plot(title='Average Monthly Time Gap')


# Time gap overall decreasing overtime, although this does not reveal much.
# The english report can be delayed as it is not a necessary service for Japanese firms to provide, although highly encouraged. 
# 

# In[7]:


#time gap vs ownership structure
# see if companies never relseased

#what companies issue ENG report and doesnt issue
#what type of companeis more likely to issue with small time gap
#later,
# any effects of issuing or not issuing (not necessarily CG perf only)
#priority: explore the time gap


# In[8]:


#how to normalize data? less spiky... would need more data to see a better pattern
# group by quarters would be more useful


# In[39]:



dumps = r'C:\Users\Derek Ho\Desktop\Uni\UW 4th year W18, W19 + TiTech\2019 Exchange TiTech\Q3Q4 Research seminar\Research Project\STATA_dumps'
# modela.to_csv('{}\{}'.format(dumps,'modela.csv'))
# exclude_industries = ['Mining','Railroad Transportation','Rubber Products',
#  'Textile Products','Trucking',
#  'Utilities - Gas','Utilities - Electric',
#  'Warehousing & Harbor Transportation']
morethanten_points = '''Air Transportation
Banks
Chemicals
Credit & Leasing
Drugs
Electric & Electronic Equipment
Foods
Insurance
Machinery
Motor Vehicles & Auto Parts
Non ferrous Metal & Metal Products
Precision Equipment
Real Estate
Securities
Services
Stone, Clay & Glass Products
Wholesale Trade'''
morethanten_points = morethanten_points.split('\n')
# use a mapping of industries to GIC sectors to better split the data


# In[9]:


dump_path = os.getcwd()+'\STATA_dumps'
file_meta = 'TOPIX_META_20200110.xlsx'
file_cg = 'TOPIX_CG_20200110.xlsx'
file_fin = 'TOPIX_FIN_20200110.xlsx'

meta = pd.read_excel(r'{}\{}'.format(dump_path,file_meta), sheet_name='main').dropna(axis=1,how='all')
cg = pd.read_excel(r'{}\{}'.format(dump_path,file_cg), sheet_name='main').dropna(axis=1,how='all')
fin = pd.read_excel(r'{}\{}'.format(dump_path,file_fin), sheet_name='main').dropna(axis=1,how='all')

tables = 'meta,cg,fin'.split(',')


# In[ ]:





# In[10]:


#%empty for each of the columns
# take out all columns with <80% empty
# if enough 100% full columns, then even better
# list(np.array((meta.Code.count() - meta.count())/meta.Code.count() * 100 < 20))
# fin.columns


# In[11]:



#take average of repeating columns
# make sure the average is only weighted by the existing values, 

# and the blanks don't skew the results
# add year column from date


# In[12]:


meta['DebtEquity_avg'] = meta[['Debt to Equity Ratio',
       'Debt to Equity Ratio.1', 'Debt to Equity Ratio.2',
       'Debt to Equity Ratio.3', 'Debt to Equity Ratio.4',
       'Debt to Equity Ratio.5', 'Debt to Equity Ratio.6',
       'Debt to Equity Ratio.7', 'Debt to Equity Ratio.8',
       'Debt to Equity Ratio.9', 'Debt to Equity Ratio.10',
       'Debt to Equity Ratio.11', 'Debt to Equity Ratio.12',
       'Debt to Equity Ratio.13']].mean(axis=1, skipna=True)
meta['EVebitda_avg'] = meta[['EV/EBITDA（Consolidated）',
                             'EV/EBITDA（Consolidated）.1']].mean(axis=1, skipna=True)
meta['market_cap_avg'] = meta['Market Capitalization(Common Stock -QUICK)'] #common stock vs listed stock.. listed ~= common*1/2
meta['year'] = meta['Date'].str.split('/', expand=True)[0]

cg['year'] = cg['Date'].str.split('/', expand=True)[0]
fin['year'] = fin['Date'].str.split('/', expand=True)[0]

meta['key'] = meta['Code'].astype(str) + "0_" + meta['year']
cg['key'] = cg['Code'].astype(str) + "0_" + cg['year']
fin['key'] = fin['Code'].astype(str) + "0_" + fin['year']


# In[13]:


# companies change to december year end, keep december data point
cg = cg.drop_duplicates(subset='key', keep='last').dropna(axis=0, subset=['key'])
meta = meta.drop_duplicates(subset='key', keep='last').dropna(axis=0, subset=['key'])
fin = fin.drop_duplicates(subset='key', keep='last').dropna(axis=0, subset=['key'])

disclosure_times['key'] = disclosure_times['company_code'].astype(str) + "_" + disclosure_times['year']


# In[14]:


# fin['current_ratio'] #divide(avg fin[['Total Current Assets', 'Total Current Assets.1']],
#                             #avg fin[[''Total Current Liabilities','Total Current Liabilities.1']])
    
# fin['BV'] = #sub(avg fin[[avg fin[['Total Current Assets', 'Total Current Assets.1']],
#                             #avg fin[[''Total Current Liabilities','Total Current Liabilities.1']])


# fin['Capex_avg']
# fin['R&D_avg']
# fin['sales_avg']


# In[15]:


# companies change to december year end, keep december data point
cg.drop_duplicates(subset='key', keep='last')


# In[16]:


#lag foreign shareholder ratio: 2015 time gap --> 2016 foreign shareholder ratio
def lag_by_group(key, value_df, lag):
    # this pandas method returns a copy of the df, with group columns assigned the key value
    df = value_df.assign(group = key) 
    return (df.sort_values(by=["key"], ascending=True)
        .set_index(["key"])
        .shift(lag) #make this -1 if want to make time_gap an independent variable instead of dependent
               ) # the parenthesis allow you to chain methods and avoid intermediate variable assignment


# In[17]:


#lag foreign shareholder ratio ==> ratio in 2014 plays factor in the time-gap for 2015
cg_grouped = cg.groupby(['Code'])
cg_list = [lag_by_group(g, cg_grouped.get_group(g), 1) for g in cg_grouped.groups.keys()]
cg_lagged = pd.concat(cg_list,axis=0).reset_index()


# Time-gap data by Industry
# 

# In[66]:


dataA = pd.merge(disclosure_times,cg_lagged, how='left', left_on='key', right_on='key')
dataA = pd.merge(dataA, meta[['DebtEquity_avg', 'EVebitda_avg','market_cap_avg', 'year', 'key']], 
                 how='inner', left_on='key', right_on='key')

dataA = dataA[dataA['year_y'].isna() == False] 
# filtering by year_y seems to make no difference after filtering out negative time gaps


# In[19]:


timegap = dataA[~dataA['time_gap'].isna()]['time_gap']


# In[20]:


# https://medium.com/@patricklcavins/using-scipys-powertransformer-3e2b792fd712
#log then normalize data

timegap = timegap.values.reshape(len(timegap),1)


# scaler = PowerTransformer('yeo-johnson')
# modela['time_gap'].loc[~modela['time_gap'].isna()] = scaler.fit_transform(timegap.values.reshape(len(timegap),1))
# plt.hist(modela['time_gap'], bins=20, density=True, log=False)
# modela[~modela['time_gap'].isna()]['time_gap'] = sc.stats.boxcox(timegap.values.reshape(len(timegap),1),0)
# modela['time_gap'].loc[~modela['time_gap'].isna()] = sc.stats.yeojohnson(timegap.values.reshape(len(timegap),1),lmbda=0.1)
pt = PowerTransformer(method='yeo-johnson', standardize=False)
d = pt.fit(timegap)
print(d.lambdas_)
calc_lambdas = d.lambdas_
d = pt.transform(timegap)
# modela['time_gap'].loc[~modela['time_gap'].isna()] = d
d
plt.hist(d, bins=20,density=True, log=True)
k2, p = sc.stats.normaltest(d,nan_policy='omit')
# timegap
print('need to find a better way to normalize the data.. how to deal with non-normal distribution data?')


# In[67]:


# cg.to_csv('{}\{}'.format(dumps,'cg_processed.csv'))
# descriptive_stats.to_csv('{}\{}'.format(dumps,'time_gap_descriptive.csv'))
dataA.columns


# Filter out the industries which have less than around 10 points for visual

# In[180]:



# modela = modela[~modela['Industry Type-Nikkei Name(E)'].isin(exclude_industries)]
# describea = modela.groupby(['year_x','Industry Type-Nikkei Name(E)']).mean()['time_gap'].unstack(level=-1).drop(columns=[0])[1:]

describea = (dataA[dataA['Industry Type-Nikkei Name(E)'].isin(morethanten_points)]
             .groupby(['year_x','Industry Type-Nikkei Name(E)'])
             .mean()['time_gap']
             .unstack(level=-1)[1:])

# describea.title('Time Delay by Industry')
ax = describea.plot(kind='line',figsize=(20,10),title='Average time gap by industry')
ax.legend(bbox_to_anchor=(1, 1))


# In[279]:


# univariate analysis 
# CURRENTLY UNUSED
# differences in mean test (t-test?)
# null hypothesis: no difference; alternate hypothesis: there is difference
# from itertools import combinations
# year_pairs = list(combinations(list(describea.index),2))
# sc.stats.normaltest()

p = [sc.stats.normaltest(dataA[dataA['year_x']== x ]['time_gap'],nan_policy='omit') 
     for x in list(describea.index)]
# normaltest_
p_values = [i[1] for i in p]
print('''
p-values for each year's data imply non-normal distribution:
2015: {}
2016: {}
2017: {}
2018: {}
2019: {}
'''.format(p_values[0],p_values[1],p_values[2],p_values[3],p_values[4]))
print(sc.stats.normaltest(dataA['time_gap'],nan_policy='omit'))


# In[70]:


#normality test of time gaps for each year
#test the difference for each year, for each industry
# sc.stats.normaltest(disclosure_times['time_gap'], axis=0,nan_policy='omit')
#group to yes and no gaps, then do a regression model, control for industry and year
dataA['has_time_gap'] = np.where(dataA['time_gap'] == 0,0, 1)

# modela['has_time_gap'] = 1 
# # modela['has_time_gap'].loc[modela['time_gap'] != 0] = 1
# modela['has_time_gap'].loc[modela['time_gap'] == 0] = 0 #.apply(lambda x: 1 if x ==0 else 0)


# In[80]:


dataA[['Code','Date','Foreign Shareholder Ratio','year_y','year_x','time_gap','key', 'has_time_gap','year','market_cap_avg']].tail(20)
# dataA.to_csv('{}\{}'.format(dumps,'dataA.csv'))


# In[208]:


###

# try a contingency table: 
# baskets of high to low foreign shareholder vs baskets of high to low time gap
# chi square test

# disribution of time gap for high vs low foreign holder
# compare distribution of the two time gap
# visual confirmation
# non-parametric statistical test for 
# testing the distributions of high vs low is different
###

#high and low bins were chosen based on the JPX's 2019 whitepaper
high_foreign = 30
med_foreign = 10
conditions = [
    (dataA['Foreign Shareholder Ratio'] >= high_foreign),
    (dataA['Foreign Shareholder Ratio'] >= med_foreign),
    (dataA['Foreign Shareholder Ratio'] < med_foreign) & (dataA['Foreign Shareholder Ratio'] > 0)]
foreign_level = ['high', 'med', 'low']
dataA['Foreign Ownership Level'] = np.select(conditions, foreign_level, default=None)

#gap bins were chosen based on intuition of how long it takes to make an investment-related decision based on 
# reading such CG reports
big_gap = 30
med_gap = 10
conditions = [
    (dataA['time_gap'] >= big_gap),
    (dataA['time_gap'] >= med_gap),
    (dataA['time_gap'] < med_gap) & (dataA['time_gap'] > 0)]
time_gap_size = ['big', 'med', 'small']
dataA['Time Gap Size'] = np.select(conditions, time_gap_size, default=None)


# In[209]:


dataA[['Time Gap Size', 'key']].groupby('Time Gap Size').count()


# In[210]:


dataA[['Foreign Ownership Level', 'key']].groupby('Foreign Ownership Level').count()


# In[211]:


dataA_contingency = (pd.crosstab(dataA['Time Gap Size'],dataA['Foreign Ownership Level'])
                     .loc[time_gap_size[ : :-1],foreign_level])

dataA_contingency


# In[212]:


chi2, p, dof, ex = sc.stats.chi2_contingency(dataA_contingency,
                                             correction=True)
print('''Pearson's Chi-square test for contingency tables: 
With {} degrees of freedom, we have a p-value of {}, and test-statistic of {}.
Thus {} is likely.'''
      .format(dof, p,chi2,'independence' if p > 0.05 else 'dependence' ))


# In[213]:


dataA[['Code','Date','Foreign Shareholder Ratio','year_y','year_x','time_gap','key', 'has_time_gap', 'Foreign Ownership Level','Time Gap Size']].tail(20)
lowforeign_timegap = dataA[dataA['Foreign Ownership Level'] =='low']['time_gap'].values
highforeign_timegap = dataA[dataA['Foreign Ownership Level'] =='high']['time_gap'].values
plt.hist(lowforeign_timegap,
         bins=10,density=True, log=True, label='low', alpha=1)
plt.hist(highforeign_timegap,
         bins=10,density=True, log=True, label='high', alpha=0.5, ls='dotted')

plt.legend(loc='upper right')
plt.title('Time Gap Histogram of High vs Low Foreign Ownership Levels')
plt.xlabel('Days between English CG Report and Japanese CG Report')
plt.show()
# plt.hist(disclosure_times['time_gap'], bins=20,density=True, log=True)


# In[214]:


# np.random.seed(0)
# sample_size = 40
# ks ,p = sc.stats.ks_2samp(np.random.choice(lowforeign_timegap,sample_size),
#                         np.random.choice(highforeign_timegap, sample_size))
ks ,p = sc.stats.ks_2samp(lowforeign_timegap,highforeign_timegap)
print('''Kolmogorov-Smirnov (KS) test for goodness of fit: 
We have a p-value of {} and test-statistic of {}.
Thus the two distributions are likely {}.'''
      .format(p,ks,'similar' if p > 0.05 else 'differnt' ))


# In[215]:


def log_plus_1(x):
    return np.log(x + 1)


# Models were chosen based on first seeing what variables would explain the variation of the time gap data.
# Variables were added and kept if R-square would increase and were statistically significant, and were removed if it resulted in multi-collinarity problems or were statistically insiginficiant. Certain variable which were not ratios such as assets, Capex, RnD, Market cap had to be re-scaled downwards to avoid the unncessary influence within the models.

# In[357]:


table = (dataA[['key','year_x','year_y','time_gap','Industry Type-Nikkei Name(E)',
                'Foreign Shareholder Ratio','has_time_gap',
                'DebtEquity_avg', 'EVebitda_avg','market_cap_avg',
                'Capital Expenditures','Research & Development Costs', 'Total Assets',
                'Outside Directors', 'Directors']]
         .dropna(axis=0,how='any')
         .rename(columns={'Industry Type-Nikkei Name(E)': 'industry', 
                          'Foreign Shareholder Ratio': 'FSratio',
                          'Capital Expenditures':'capex',
                          'Research & Development Costs' : 'RnD',
                          'Total Assets': 'assets'}))

table['outsidedirector_ratio'] = (table['Outside Directors']/ table['Directors'])* 100
# industry_ind = pd.get_dummies(table['industry'])
# year_ind = pd.get_dummies(table['year_x'])
# table = pd.concat([table, industry_ind, year_ind], axis=1)


# table.tail(20)
# model_a = smf.mixedlm('time_gap ~ FSratio', table, groups=table['industry']+table['year_x'])
model_timegap1 = (
    smf.ols(formula='''log_plus_1(time_gap) ~ FSratio + 
                                              outsidedirector_ratio + 
                                              np.divide(RnD,10000) +                                                 
                                              C(industry) + 
                                              C(year_x)''', 
            data=table).fit())
print(model_timegap1.summary())
# variables tested and not used:
# EVebitda_avg, DebtEquity_avg, np.log(np.divide(market_cap_avg,100000000000)), np.log(np.divide(assets,10000)), , np.divide(RnD,10000)


# After controlling for Assets, capex, the proportion of outside directors, and industry and time fixed effects, we can see an increasae in percentage of foreign shareholders in a company the previous year should decreased the time gap. This is statistically significant and the 5% level. Other control variables were added (R&D spending, Debt Equity Average, EV/EBITDA average, Market Capitalization Average) but were found to not contribute any improvement in the describing the variation in time gap data. In other words, the R-square value was little changed and they had no statistically significant relationships with the time gap data.

# In[99]:


#do the test in the opposite direction
cg_grouped = cg.groupby(['Code'])
cg_list_forward = [lag_by_group(g, cg_grouped.get_group(g), -1) for g in cg_grouped.groups.keys()]
cg_forward = pd.concat(cg_list_forward,axis=0).reset_index()

dataB = pd.merge(disclosure_times,cg_forward, how='left',left_on='key', right_on='key')
dataB = pd.merge(dataB, meta[['DebtEquity_avg', 'EVebitda_avg','market_cap_avg', 'year', 'key']],
                 how='inner',left_on='key', right_on='key')
dataB = dataB[dataB['year_y'].isna()==False]
dataB['has_time_gap'] = np.where(dataB['time_gap'] != 0,1, 0)

## for some reason, time lag is not present?
dataB[['Code','Date','Foreign Shareholder Ratio','year_y','year_x','time_gap','key', 'has_time_gap','year','market_cap_avg']].tail(20)


# In[359]:


table = (dataB[['key','year_x','year_y','time_gap','Industry Type-Nikkei Name(E)',
                'Foreign Shareholder Ratio','has_time_gap',
                'DebtEquity_avg', 'EVebitda_avg','market_cap_avg',
                'Capital Expenditures','Research & Development Costs', 'Total Assets',
                'Outside Directors', 'Directors']]
         .dropna(axis=0,how='any')
         .rename(columns={'Industry Type-Nikkei Name(E)': 'industry', 
                          'Foreign Shareholder Ratio': 'FSratio',
                          'Capital Expenditures':'capex',
                          'Research & Development Costs' : 'RnD',
                          'Total Assets': 'assets'}))

table['outsidedirector_ratio'] = (table['Outside Directors']/ table['Directors'])* 100
# industry_ind = pd.get_dummies(table['industry'])
# year_ind = pd.get_dummies(table['year_x'])
# table = pd.concat([table, industry_ind, year_ind], axis=1)


# table.tail(20)
# model_a = smf.mixedlm('time_gap ~ FSratio', table, groups=table['industry']+table['year_x'])

model_timegap2 = (
    smf.ols(formula='''FSratio ~ log_plus_1(time_gap)+
                                              outsidedirector_ratio + 
                                              np.log(np.divide(assets,10000)) + 
                                              np.log(np.divide(market_cap_avg,100000000000)) +
                                              np.log(np.divide(capex,10000)) + 
                                              np.divide(RnD,10000) + 
                                              
                                              
                                              C(industry) + 
                                              C(year_x)''', 
            data=table).fit())
print(model_timegap2.summary())
# variables tested and not used:
# EVebitda_avg, DebtEquity_avg, has_time_gap


# Reversing the test, we find a decreased in the time gap does not increase the perentage of foreign shareholders in a company. Indeed this is not statistically significant relationship (at the 5% level) and makes intuitive sense. Foreign investors have many other factors to consider before investing in a firm, where as foreign shareholders, who have already decided to invest in the firm, would likely want to know how the firm is performing regularly. And so, would demand an english CG report. The firm, who must consider the needs of it's investors are more likely to comply to the demand of it's foreign investors for an english version of the CG report. If a firm wanted to increase the percentage of foreign investment in itself, there are more effective and assertive than producing an english version of their CG reports. 

# In[ ]:


### REMAINING THINGS TO DO:
# maybe delay the time gap by two years instead of one year..
# try adding in company returns for the year
# email professor with code files, how-to, libraries to have installed, and research summary
# write proposal, email this to professor as well

### things not done:
contingency cube


# In[ ]:


#join the three tables based on company code and date 
#create dependent variables data
# do descriptive stats on them
data = merge().merge()
data['BVtoequity']#market valuation vs company valuation
data['TobinsQ'] #


# Considering the use of R&D and CAPEX, as these are proxies for corporate risk-taking. 
# Paper from India focuses on how these factors change as a result of a CGR.
# The paper also hypothesis the risk-taking activities would shift for otherwise conservaitve firms because of concentrated ownership
# .. If I recall correctly, firms in Japan also have concentrated ownership. Furtherore, Japanese firms are characterized by the "quiet life", as postulated by Professor Kotaru Inoue's paper. Further risk-taking (in the form of CAPEX and R&D) could be seen as a proxy for better corproate governance
# 
# The paper also stipulates increase in the number of independnent board members

# In[ ]:


fin.columns


# In[ ]:


#industry analysis
industry = meta[['Code','Industry Type-Nikkei Name(E)']]
#distribution of industries
#create categorical variable(indicator variable) for industry
# time_gap ~ industries + company_fixed + time_fixed
# time_gap ~ Market_cap + industries

#debt_equity, EVEBITDA, industry, marketcap
#see summary data (R2 value)


# In[ ]:


# summary of data
plt.style.use('ggplot')
# Histogram of the height
df.Height.plot(kind='hist',color='purple',edgecolor='black',figsize=(10,7))
plt.title('Distribution of Height', size=24)
plt.xlabel('Height (inches)', size=18)
plt.ylabel('Frequency', size=18)

# Histogram of the weight
df.Weight.plot(kind='hist',color='purple',edgecolor='black',figsize=(10,7))
plt.title('Distribution of Weight', size=24)
plt.xlabel('Weight (pounds)', size=18)
plt.ylabel('Frequency', size=18);


# In[ ]:


#descriptive statistics
#pd.describe


# In[ ]:


from sklearn.linear_model import LinearRegression

df_males = df[df['Gender']=='Male']

# Create linear regression object.
lr_males= LinearRegression()

# Fit linear regression.
lr_males.fit(df_males[['Height']], df_males['Weight'])

# Get the slope and intercept of the line best fit.
print(lr_males.intercept_)
# -224.49884070545772

print(lr_males.coef_)
# 5.96177381
from sklearn.linear_model import LinearRegression


# In[ ]:


# Create linear regression object.
mlr= LinearRegression()

# Fit linear regression.
mlr.fit(df_dummy[['Height','Gender']], df_dummy['Weight'])

# Get the slope and intercept of the line best fit.
print(mlr.intercept_)
# -244.92350252069903

print(mlr.coef_)
# [ 5.97694123 19.37771052]


# In[ ]:


#calculate time gaps
#see descriptive stats of time gaps
#classify time gaps as small or large


# In[ ]:


#panel data regression is the goal?
#preliminary: any of the dependents change with release of english CG reports
#

