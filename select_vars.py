#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import sklearn.preprocessing as skp
import sklearn.metrics as sm

from sklearn.ensemble import RandomForestClassifier as rfc

import seaborn as sns
import lightgbm as lgbm

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.options.display.float_format = '{:.2f}'.format

from dataminer.connector import BigRed
from sklearn.model_selection import train_test_split
import json
import gc
gc.collect()
cred = json.load(open("credentails.json"))

session=BigRed(username=cred['zid'], password=cred['pass'])


# ## Variable selection

# In[10]:


existing_features = ['shp_req_q', 'availableqty', 'availableqtywithoutthreshold', 'days_sold_year', 'sales_year', 'units_year', 'days_sold_1m', 'days_sold_1m_norm', 'units_1m', 'days_sold_7d', 'units_7d', 'sales_1m', 'days_sold_3m', 'units_3m', 'sales_3m',
                     'sales_3m_norm', 'days_since_last_sale_both', 'item_year_sales', 'no_of_stores_sold', 'store_year_sales', 'store_year_units', 'days_since_last_sale', 'days_since_first_sale', 'days_sold_norm', 'avg_eoh_3d', 'avg_eoh_2d',
                     'sales_year_norm', 'units_year_norm', 'retl_a', 'avg_eoh_7d', 'avg_eoh_q', 'avg_eoh_30d', 'days_avail', 'eoh_q_lt_3', 'perc_eoh_lt_3', 'ipi', 'store_fulfill_units', 'store_inf_units', 'store_inf_rate', 'item_fulfill_units',
                     'item_inf_units', 'item_inf_rate', 'dept_clas_fulfill_units', 'dept_clas_inf_units', 'dept_clas_inf_rate', 'total_oh', 'nbr_str', 'avg_oh_network', 'si_inf_rate', 'weight', 'back_room_quantity', 'si_3_inf_rate', 'si_7_inf_rate',
                     'si_14_inf_rate', 'si_30_inf_rate', 'i_3_inf_rate', 'i_7_inf_rate', 'i_14_inf_rate', 'i_30_inf_rate', 's_3_inf_rate', 's_7_inf_rate', 's_14_inf_rate', 's_30_inf_rate', 'sdc_3_inf_rate', 'sdc_7_inf_rate', 'sdc_14_inf_rate',
                     'dc_3_inf_rate', 'dc_7_inf_rate', 'dc_14_inf_rate', 'dc_30_inf_rate', 'research_evnt_exists_3days', 'any_evnt_exists_3days', 'research_evnt_exists_7days', 'any_evnt_exists_7days', 'si_3_fulfill', 'i_3_fulfill', 's_3_fulfill',
                     'sdc_3_fulfill', 'dc_3_fulfill', 'boh_q', 'eoh_q', '7d_avg_boh_q', '7d_avg_eoh_q', '7d_boh_2_sls_rate', '7d_eoh_2_sls_rate', '7d_sell_thru_rate',
                     '7d_inv_trnovr_rate', '7d_days_to_sell_inv', '7d_actual_oh_avg', 'repln_q', '7d_repln_rate', '7d_dc_high_repln_rate',
                     '7d_doh', 'shp_oos_eoh_q', '7d_min_shp_oos_eoh_q', 'sls_forcast_q', 'threshold', 'allowance', 'release_qty', 'received_hr']


existing_exclusion_features = ['store_fulfill_units', 'store_inf_units', 'store_inf_rate', 'item_fulfill_units', 'item_inf_units', 'item_inf_rate', 'dept_clas_fulfill_units', 'dept_clas_inf_units',
                               'dept_clas_inf_rate', 'atp_units']
dep_var = [x for x in existing_features if x not in existing_exclusion_features]


# In[11]:


# # select_var = ['7d_actual_oh_avg',
# # '7d_avg_boh_q',
# # '7d_avg_eoh_q',
# # '7d_dc_high_repln_rate',
# # '7d_doh',
# # '7d_eoh_2_sls_rate',
# # '7d_inv_trnovr_rate',
# # '7d_repln_rate',
# # '7d_sell_thru_rate',
# # 'allowance',
# # 'availableqty',
# # 'availableqtywithoutthreshold',
# # 'avg_eoh_2d',
# # 'avg_eoh_30d',
# # 'avg_eoh_3d',
# # 'avg_eoh_7d',
# # 'avg_eoh_q',
# # 'avg_oh_network',
# # 'back_room_quantity',
# # 'boh_q',
# # 'days_since_first_sale',
# # 'days_since_last_sale_both',
# # 'days_sold_1m_norm',
# # 'dc_3_fulfill',
# # 'eoh_q',
# # 'eoh_q_lt_3',
# # 'i_3_fulfill',
# # 'i_30_inf_rate',
# # 'ipi',
# # 'item_year_sales',
# # 'nbr_str',
# # 'no_of_stores_sold',
# # 'perc_eoh_lt_3',
# # 'received_hr',
# # 'release_qty',
# # 'repln_q',
# # 'retl_a',
# # 's_3_fulfill',
# # 'sales_3m',
# # 'sales_3m_norm',
# # 'sales_year',
# # 'sdc_14_inf_rate',
# # 'sdc_7_inf_rate',
# # 'shp_oos_eoh_q',
# # 'shp_req_q',
# # 'si_14_inf_rate',
# # 'si_3_fulfill',
# # 'si_3_inf_rate',
# # 'si_30_inf_rate',
# # 'si_7_inf_rate',
# # 'si_inf_rate',
# # 'sls_forcast_q',
# # 'store_year_sales',
# # 'store_year_units',
# # 'threshold',
# # 'total_oh',
# # 'units_year_norm',
# # 'allowance_c','availableqty_c','bkrm_qt_c',
# # 'weight']

select_var = ['back_room_quantity',
'allowance',
'i_30_inf_rate',
'availableqty',
'si_inf_rate',
'threshold',
'avg_eoh_30d',
'sdc_14_inf_rate',
'i_3_fulfill',
'received_hr',
'sdc_7_inf_rate',
'days_since_last_sale_both',
'dc_3_fulfill',
'weight',
'avg_eoh_q',
'retl_a',
'days_since_first_sale',
'store_year_units',
'units_year_norm',
'sls_forcast_q',
'7d_repln_rate',
'avg_oh_network',
'nbr_str',
'perc_eoh_lt_3',
'store_year_sales',
'days_sold_1m_norm',
'sales_3m',
'availableqtywithoutthreshold',
's_3_fulfill',
'eoh_q_lt_3',
'si_30_inf_rate',
'si_14_inf_rate',
'ipi',
'7d_doh',
'7d_avg_eoh_q',
'7d_inv_trnovr_rate',
'no_of_stores_sold',
'sales_year',
'item_year_sales',
'total_oh',
'sales_3m_norm',
'7d_actual_oh_avg',
'7d_dc_high_repln_rate',
'7d_avg_boh_q',
'si_3_inf_rate',
'eoh_q',
'avg_eoh_7d',
'7d_eoh_2_sls_rate']


# In[12]:


var = dep_var + ['inf_flag']


# ## Functions

# In[5]:


import warnings
warnings.filterwarnings('ignore')


# def var_creat(df):
#     df['allowance_c'] = pd.cut(df['allowance'],[0,3.5,7,10,13,16], include_lowest=True, labels=False)
#     df['availableqty_c'] = pd.cut(df['availableqty'],[0,2,4,6,8,10,15], include_lowest=True, labels=False)
#     df['bkrm_qt_c'] = pd.cut(june_df2['back_room_quantity'],[0,1,2,4,6,8,16,32,54,90,130,1167], include_lowest=True, labels=False)
#     return df


def Preprep(sample_df):
    df = sample_df.copy()
    columns = [x.split("s.")[1] for x in df.columns]
    df.columns = columns
    
    df = df.fillna(0)
    df['availableqtywithoutthreshold'] = df['onhand']
    df['allowance'] = df['availableqtywithoutthreshold'] - df['shp_req_q']
    df['eoh_q'] = df['eoh_q'].astype('float')
    df['release_qty'] = df['eoh_q'] - df['availableqtywithoutthreshold']
    df['inf_flag'] = np.where(df['inf_q'] > 0, 1, 0)
    df['received_hr'] = df['received_ts'].apply(lambda x: x.hour)
    df['received_hr_c'] = pd.cut(df['received_hr'], bins=4, labels=False)
    
    for col in dep_var:
        df[col] = df[col].astype('float')
    
    df['allowance_c'] = pd.cut(df['allowance'],[0,3.5,7,10,13,16], include_lowest=True, labels=False)
    df['availableqty_c'] = pd.cut(df['availableqty'],[0,2,4,6,8,10,15], include_lowest=True, labels=False)
    df['bkrm_qt_c'] = pd.cut(df['back_room_quantity'],[0,1,2,4,6,8,16,32,54,90,130,1167], include_lowest=True, labels=False)
    
    return df


def coef_var(x):
    return (np.std(x)/np.mean(x))*100


def summary(df):
    summary1 = df.groupby('inf_flag').aggregate([np.mean, np.std]).T.unstack()
    summary2 = df.aggregate(['mean', 'std']).T
    summary = summary1.merge(summary2, right_index=True, left_index=True)
    summary.columns = ['0_mean', '0_std', '1_mean', '1_std', 'mean', 'std']
    summary['coef_0'] = summary['0_std']/summary['0_mean']
    summary['coef_1'] = summary['1_std']/summary['1_mean']
    summary['coef_all'] = summary['std']/summary['mean']
    return summary


def outlier_treatment(col):
    '''
    returns the upper limit for outliers
    '''
    q1, q3 = np.quantile(col, [.25, .75])
    iqr = q3 - q1
    return q3 + 1.5*iqr


def outlier_treatment_df(main_df, depvar=dep_var):
    '''
    input : main_df, dep_var
    return's : df, iqr_list
    '''
    df = main_df.copy()
    iqr_list = {}
    for col in depvar:
        iqr_list[col] = outlier_treatment(df[col])
    for col in depvar:
        df['upper'] = iqr_list[col]
        df[col] = [y if x > y else x for x, y in zip(df[col], df['upper'])]
        df = df.drop(columns=['upper'])
    return df, iqr_list


def treatment(data, dep_var):
    dt = data.copy()
#     rate_cols = ['availableqtywithoutthreshold', 'sales_1m', 'si_3_inf_rate', 'si_7_inf_rate', 'si_14_inf_rate', 'si_30_inf_rate',
#                  'i_3_inf_rate', 'i_7_inf_rate', 'i_14_inf_rate', 'i_30_inf_rate',
#                  'sdc_3_inf_rate', 'sdc_7_inf_rate', 'sdc_14_inf_rate',
#                  '7d_inv_trnovr_rate', '7d_days_to_sell_inv', 'shp_oos_eoh_q',
#                  '7d_min_shp_oos_eoh_q', 'allowance', 'release_qty']
    
    for x in dep_var:
        dt[x] = np.where(dt[x] < 0, 0, dt[x])
        dt[x] = np.where(dt[x] < 0, 0, dt[x])

    summary_original = summary(data)
    treated_df, iqr_list = outlier_treatment_df(dt, dep_var)
    
    summary_treated = summary(treated_df)

    temp = summary_original[['coef_all']].merge(
        summary_treated[['coef_all']], right_index=True, left_index=True)
    temp['coeff_dif'] = np.where(temp.coef_all_x > temp.coef_all_y, 0, 1)
    var_for_treat = list(temp[temp.coeff_dif == 0].index)

    for col in var_for_treat:
        dt['upper'] = iqr_list[col]
        dt[col] = [y if x > y else x for x, y in zip(dt[col], dt['upper'])]
        dt = dt.drop(columns=['upper'])
    return dt


# In[7]:


# query = ''' select s.* from z0019dc.threshold_by_fulfillment_type_base_data_stg6 TABLESAMPLE(0.75 PERCENT) s
#                     where fulfillment_type_id = 3 and
#                    month(local_attempted_d) = {0} and
#                    mdse_grp_n in ("HARDLINES")
#                    '''


# query = '''
# with t1 as (
# select s.*,
# floor(rand() * 10 + 1) as random_no
# from z0019dc.threshold_by_fulfillment_type_base_data_stg6  as s
#                     where fulfillment_type_id = 3 and
#                    month(local_attempted_d) = {0} and
#                    mdse_grp_n in ("HARDLINES")
#                    )
# select s.*
# from t1 as s
# where s.random_no <= {1}
#                    '''

query = ''' 
with t1 as (
select s.*
from z0019dc.threshold_by_fulfillment_type_base_data_stg6  as s
                    where fulfillment_type_id = 3 and
                   month(local_attempted_d) = {0} and
                   mdse_grp_n in ("HARDLINES")
                   )
select s.* 
from t1 as s
where floor(rand()*100) <= {1}
distribute by rand()
sort by rand()
                   '''


query_gt0 = ''' 
with t1 as (
select s.*
from z0019dc.threshold_by_fulfillment_type_base_data_stg6  as s
                    where fulfillment_type_id = 3 and
                   month(local_attempted_d) = {0} and
                   mdse_grp_n in ("HARDLINES") and
                   inf_q > 0
                   )
select s.* 
from t1 as s
where floor(rand()*100) <= {1}
distribute by rand()
sort by rand()
                   '''


query_lt0 = ''' 
with t1 as (
select s.*
from z0019dc.threshold_by_fulfillment_type_base_data_stg6  as s
                    where fulfillment_type_id = 3 and
                   month(local_attempted_d) = {0} and
                   mdse_grp_n in ("HARDLINES") and
                   inf_q <= 0
                   )
select s.* 
from t1 as s
where floor(rand()*100) <= {1}
distribute by rand()
sort by rand()
                   '''


month = [6, 7, 8, 9]
month_name = ['June', 'July', 'Aug', 'Sep']


mdse_grp_n = ['APPAREL_ACCESS', 'BEAUTY_COSMETICS',
              'ESSENTIALS', 'FOOD_BEVERAGE', 'HARDLINES']


# ## Data Import

# ### June_df

# In[13]:


temp=session.query(query_gt0.format(6,50))
# temp = temp.sample(100000)


# In[14]:


june_df1 = Preprep(temp)


# In[ ]:





# In[15]:


temp=session.query(query_lt0.format(6,90))
temp = temp.sample(100000)

june_df2 = Preprep(temp)


# In[16]:


june_df = june_df1.append(june_df2,ignore_index=True)
june_df = june_df.reset_index(drop=True)
june_df = june_df[var]
print(june_df['inf_flag'].sum())
print(june_df.shape)


# In[22]:


del june_df1, june_df2, temp


# ### July df import

# In[23]:


july_df = Preprep(session.query(query.format(7,10)))
july_df = july_df[var]
print(july_df.shape)


# In[ ]:





# ### June df Original import

# In[24]:


june_df2 = Preprep(session.query(query.format(6,10)))
june_df2 = june_df2[var]


# #### Treatment

# In[25]:



june_df2 = treatment(june_df2,dep_var)


# In[26]:


june_df2['inf_flag'].value_counts()


# In[27]:


def var_car(df):
    df['allowance_c'] = pd.cut(df['allowance'],[0,3.5,7,10,13,16], include_lowest=True, labels=False)
    df['availableqty_c'] = pd.cut(df['availableqty'],[0,2,4,6,8,10,15], include_lowest=True, labels=False)
    df['bkrm_qt_c'] = pd.cut(df['back_room_quantity'],[0,1,2,4,6,8,16,32,54,90,130,1167], include_lowest=True, labels=False)
    return df


# In[28]:


june_df2 = var_car(june_df2)
june_df = var_car(june_df)
july_df = var_car(july_df)


# In[ ]:





# ## Data summary and treatment

# In[27]:





# In[29]:


june_pre_summ = summary(june_df)
# june_pre_summ.head()


# ### Negative values removal

# In[30]:


summarydf = june_df.describe().T


rate_cols = ['availableqtywithoutthreshold', 'sales_1m', 'si_3_inf_rate', 'si_7_inf_rate', 'si_14_inf_rate', 'si_30_inf_rate',
             'i_3_inf_rate', 'i_7_inf_rate', 'i_14_inf_rate', 'i_30_inf_rate',
             'sdc_3_inf_rate', 'sdc_7_inf_rate', 'sdc_14_inf_rate',
             '7d_inv_trnovr_rate', '7d_days_to_sell_inv', 'shp_oos_eoh_q',
             '7d_min_shp_oos_eoh_q', 'allowance', 'release_qty']

for x in rate_cols:
    june_df[x] = np.where(june_df[x] < 0,0,june_df[x])
    july_df[x] = np.where(july_df[x] < 0,0,july_df[x])

for x in rate_cols:
    june_df2[x] = np.where(june_df2[x] < 0,0,june_df2[x])


# In[195]:





# In[31]:


treated_df , iqr_list = outlier_treatment_df(june_df, dep_var)

print("***** Summary Before treatment *****")
summary_original = summary(june_df)
# summary_original.head(7)

print("***** Summary After treatment *****")
summary_treated = summary(treated_df)
# summary_treated.head(7)


# In[32]:


temp = summary_original[['coef_all']].merge(
    summary_treated[['coef_all']], right_index=True, left_index=True)

temp['coeff_dif'] = np.where(temp.coef_all_x > temp.coef_all_y, 0, 1)

# temp.head()

var_for_treat = list(temp[temp.coeff_dif == 0].index)


# In[33]:


treated_df , iqr_list = outlier_treatment_df(june_df, var_for_treat)

for col in var_for_treat:
        july_df['upper'] = iqr_list[col]
        july_df[col] = [y if x>y else x for x,y in zip(july_df[col],july_df['upper'])]
        july_df = july_df.drop(columns=['upper'])

for col in var_for_treat:
        june_df2['upper'] = iqr_list[col]
        june_df2[col] = [y if x>y else x for x,y in zip(june_df2[col],june_df2['upper'])]
        june_df2 = june_df2.drop(columns=['upper'])


# ### Graph

# In[194]:


import matplotlib.pyplot as plt
import seaborn as sns

# plt.hist(june_df2[june_df2.inf_flag==1]['availableqty'], bins=12,color='green')
sns.boxplot(y=june_df2['availableqty'], x=june_df2['inf_flag'])
plt.title('availableqty')
plt.show()

sns.boxplot(y=june_df2['allowance'], x=june_df2['inf_flag'])
plt.title('allowance')
plt.show()


# In[705]:


pd.cut(june_df2['availableqty'],[0,2,4,6,8,10,15], include_lowest=True, labels=False,retbins=True)


# In[709]:


june_df2['back_room_quantity'][june_df2.back_room_quantity >0].describe()


# In[716]:


sns.boxplot(y=june_df2['back_room_quantity'][(june_df2.back_room_quantity >0) & (june_df2.back_room_quantity <20)], x=june_df2['inf_flag'])
plt.title('back_room_quantity')
plt.show()


# In[726]:


june_df2[['inf_flag','back_room_quantity']][(june_df2.back_room_quantity >0) & (june_df2.back_room_quantity <20)].groupby('inf_flag').describe(percentiles=[.25,.35,.5,.75,.9,.95,.99]).T


# In[724]:


june_df2[['inf_flag','back_room_quantity']][june_df2.back_room_quantity >0].groupby('inf_flag').describe(percentiles=[.25,.5,.75,.9,.95,.99]).T


# In[728]:


pd.cut(june_df2['back_room_quantity'],[0,1,2,4,6,8,16,32,54,90,130,1167], include_lowest=True, labels=False,retbins=True)


# In[707]:


pd.cut(june_df2['allowance'],[0,3.5,7,10,13,16], include_lowest=True, labels=False,retbins=True)


# ## Model deployment original data

# In[51]:


june_train, june_test, june_y_train, june_y_test = train_test_split(june_df2[select_var],june_df2['inf_flag'], train_size=0.7)


# In[52]:


sum(june_y_train)


# ### LGBM Model(Orignal)

# In[54]:


lgbm_1 = lgbm.LGBMClassifier(n_estimators = 300, max_depth=20,learning_rate=0.05,subsample=.5,reg_alpha=0.2,reg_lambda=0.2                             ,class_weight={0:.8,1:1.5},num_leaves=400,importance_type='gain'
                             ,boosting_type="dart",n_jobs = -1)

model_original = lgbm_1.fit(june_train,june_y_train)


# In[56]:


train_prediction = model_original.predict(june_train)
test_prediction = model_original.predict(june_test)

print("*****. TRAIN DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(train_prediction, june_y_train))


print("*****. VALIDATION DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(test_prediction, june_y_test))

test_prediction = model_original.predict(july_df[select_var])


print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ### Select_var

# In[41]:


select_var = [
# 'back_room_quantity',
'allowance',
'i_30_inf_rate',
'availableqty',
'si_inf_rate',
# 'threshold',
'avg_eoh_30d',
'sdc_14_inf_rate',
'i_3_fulfill',
'received_hr_c',
'sdc_7_inf_rate',
'days_since_last_sale_both',
'dc_3_fulfill',
'weight',
'avg_eoh_q',
'retl_a',
'days_since_first_sale',
'store_year_units',
'units_year_norm',
'sls_forcast_q',
'7d_repln_rate',
'avg_oh_network',
'nbr_str',
'perc_eoh_lt_3',
'store_year_sales',
'days_sold_1m_norm',
'sales_3m',
'availableqtywithoutthreshold',
's_3_fulfill',
'eoh_q_lt_3',
'si_30_inf_rate',
'si_14_inf_rate',
'ipi',
'7d_doh',
'7d_avg_eoh_q',
'7d_inv_trnovr_rate',
'no_of_stores_sold',
'sales_year',
'item_year_sales',
'total_oh',
'sales_3m_norm',
'7d_actual_oh_avg',
'7d_dc_high_repln_rate',
'7d_avg_boh_q',
'si_3_inf_rate',
'eoh_q',
'avg_eoh_7d',
'7d_eoh_2_sls_rate']


# In[40]:


len(select_var2)


# In[34]:


june_train, june_test, june_y_train, june_y_test = train_test_split(june_df2[select_var],june_df2['inf_flag'], train_size=0.7)


# #### M_Original (m_o1)

# In[35]:


lgbm_2 = lgbm.LGBMClassifier(n_estimators = 1000, max_depth=20,learning_rate=0.05,subsample=.5,reg_alpha=0.2,reg_lambda=0.2)

m_o1 = lgbm_2.fit(june_train,june_y_train)

train_prediction = m_o1.predict(june_train)
test_prediction = m_o1.predict(june_test)


# In[36]:


print("*****. TRAIN DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(train_prediction, june_y_train))


print("*****. VALIDATION DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(test_prediction, june_y_test))

test_prediction = m_o1.predict(july_df[select_var])


print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ## Model deployment resample data

# ### Train-Test-Split

# ### LGBM

# In[59]:


x_train, x_test, y_train, y_test = train_test_split(treated_df[select_var],treated_df['inf_flag'], train_size=0.7)


# In[60]:


### Testing model old and examin if we have improvement in data 
lgbm_base = lgbm.LGBMClassifier(n_estimators = 2000, random_state = 1234, n_jobs = 5)

base_model = lgbm_base.fit(x_train,y_train)

train_prediction = base_model.predict(x_train)
test_prediction = base_model.predict(x_test)


# In[61]:


print("*****. TRAIN DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


# In[62]:


print("*****. VALIDATION DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


# In[63]:


test_prediction = base_model.predict(july_df[select_var])


print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[64]:


test_prediction = base_model.predict(june_df2[select_var])


print("*****. June_df unbalanced TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(june_df2['inf_flag'], test_prediction))


# In[ ]:





# ### After model tuning (M2)

# In[65]:


lgbm_1 = lgbm.LGBMClassifier(n_estimators = 250, n_jobs = -1, max_depth=20,class_weight={0:1.5,1:1},num_leaves=60, importance_type='gain'
                             ,boosting_type="dart" )

model_1 = lgbm_1.fit(x_train,y_train)


# In[66]:


train_prediction = model_1.predict(x_train)
test_prediction = model_1.predict(x_test)

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_1.predict(july_df[select_var])
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[67]:


lgbm_1 = lgbm.LGBMClassifier(n_estimators = 200, n_jobs = -1, max_depth=10,class_weight={0:2.5,1:1},num_leaves=50, importance_type='gain'
                             ,boosting_type="dart" )

model_1 = lgbm_1.fit(x_train,y_train)


# In[68]:


train_prediction = model_1.predict(x_train)
test_prediction = model_1.predict(x_test)

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_1.predict(july_df[select_var])
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[69]:


test_prediction = model_1.predict(june_df2[select_var])
print("*****. *****. June_df unbalanced TEST DATA:: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(june_df2['inf_flag'], test_prediction))


# ### LGBM Select Vars (model_2)

# In[71]:


lgbm_2 = lgbm.LGBMClassifier(n_estimators = 200, n_jobs = -1, max_depth=10,class_weight={0:2.5,1:1},num_leaves=50, importance_type='gain'
                             ,boosting_type="dart" )
## june_train, june_test, june_y_train, june_y_test
model_2 = lgbm_2.fit(x_train[select_var],y_train)


# In[72]:


train_prediction = model_2.predict(x_train[select_var])
test_prediction = model_2.predict(x_test[select_var])

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_2.predict(july_df[select_var])
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[ ]:


from sklearn.support


# ### Random Forest

# In[ ]:





# In[145]:


from sklearn.ensemble import RandomForestClassifier as rfc


# In[73]:


rfc_1 = rfc(n_estimators = 300, n_jobs = -1, max_depth=15,class_weight={0:2,1:1},min_samples_leaf=2)

rfc_model = rfc_1.fit(x_train,y_train)


# In[74]:


train_prediction = rfc_model.predict(x_train)
test_prediction = rfc_model.predict(x_test)

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = rfc_model.predict(july_df[dep_var])
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[75]:


test_prediction = rfc_model.predict(june_df2[dep_var])
print("*****. June_df unbalanced TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(june_df2['inf_flag'], test_prediction))


# ## Catboost

# In[74]:


from catboost import CatBoostClassifier as cat


# ### Original Data

# #### Model: cat_m1

# In[75]:


cat_m1 = cat(random_state=0,
             #              scale_pos_weight=5,
             class_weights={0: 1, 1: 1.5},
#              auto_class_weights='SqrtBalanced',
             iterations=1000,
             depth=10,
             learning_rate=0.05,
             custom_loss=['AUC', 'F1'],
             eval_metric= 'F1:use_weights=true',
             verbose=False,
             use_best_model=True)
cat_m1.fit(june_train, june_y_train, eval_set=(
    june_test, june_y_test),
        early_stopping_rounds=600,
    plot=True)


# In[76]:


train_prediction = cat_m1.predict(june_train)
test_prediction = cat_m1.predict(june_test)

print("Model:","cat_m1:")

print("*****. TRAIN DATA: CatBoost MODEL2 (Hardlines) ********")
print(sm.classification_report(train_prediction, june_y_train))


print("*****. VALIDATION DATA: CatBoost MODEL2 (Hardlines) ********")
print(sm.classification_report(test_prediction, june_y_test))

test_prediction = cat_m1.predict(july_df[select_var])


print("*****. July TEST DATA: CatBoost MODEL2 (Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# #### Model: cat_m2

# In[77]:


cat_m2 = cat(random_state=0,
             auto_class_weights='SqrtBalanced',
             iterations=243,
             depth=10,
             learning_rate=0.05,
             custom_loss=['AUC', 'F1'],
             eval_metric='F1', #'F1:use_weights=true',
             verbose=False,
             use_best_model=True)
cat_m2.fit(june_train, june_y_train, eval_set=(
    june_test, june_y_test),
#     early_stopping_rounds=600,
    plot=True)


# In[78]:


train_prediction = cat_m2.predict(june_train)
test_prediction = cat_m2.predict(june_test)

print("Model:","cat_m2")

print("*****. TRAIN DATA: CatBoost MODEL3 (Hardlines) ********")
print(sm.classification_report(train_prediction, june_y_train))


print("*****. VALIDATION DATA: CatBoost MODEL3 (Hardlines) ********")
print(sm.classification_report(test_prediction, june_y_test))

test_prediction = cat_m2.predict(july_df[select_var])

print("*****. July TEST DATA: CatBoost MODEL3 (Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[141]:


pd.DataFrame({'Vars':select_var,'importance':cat_m2.feature_importances_}).set_index('Vars').sort_values(by='importance').plot(kind='barh',figsize=(10,18))
plt.show()


# In[140]:


pd.DataFrame({'Vars':select_var,'importance':cat_m1.feature_importances_}).set_index('Vars').sort_values(by='importance')


# In[142]:


pd.DataFrame({'Vars':select_var,'importance':cat_m2.feature_importances_}).set_index('Vars').sort_values(by='importance')


# In[81]:


train_pool = Pool(june_test, june_y_test)
train_pool_slice = train_pool.slice([5,10])

importance_df = cat_m1.get_feature_importance(train_pool_slice,
                                              type='PredictionDiff',
                                              prettified=True).set_index('Feature Id').\
sort_values(by='Importances')
importance_df.plot(kind='barh', figsize=(12, 30))
plt.show()


# In[500]:


importance_df.tail()


# #### Model: cat_m3

# In[85]:


cat_m3 = cat(random_state=0,
             scale_pos_weight=3,
#              auto_class_weights='SqrtBalanced',
             iterations=1000,
             depth=7,
             learning_rate=0.1,
             custom_loss=['AUC', 'F1'],
             eval_metric= 'F1:use_weights=true',
             verbose=False,
             bagging_temperature=.5,
             use_best_model=True)
cat_m3.fit(june_train, june_y_train, eval_set=(
    july_df[select_var], july_df['inf_flag']),
    early_stopping_rounds=220,
    plot=True)



train_prediction = cat_m3.predict(june_train)
test_prediction = cat_m3.predict(june_test)

print("*****. TRAIN DATA: CatBoost MODEL spw=3 (Hardlines) ********")
print(sm.classification_report(train_prediction, june_y_train))


print("*****. VALIDATION DATA: CatBoost MODEL spw=3(Hardlines) ********")
print(sm.classification_report(test_prediction, june_y_test))

test_prediction = cat_m3.predict(july_df[select_var])


print("*****. July TEST DATA: CatBoost MODEL spw=3(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[143]:


pd.DataFrame({'Vars':select_var,'importance':cat_m3.feature_importances_}).set_index('Vars').sort_values(by='importance')


# ### Resample data

# #### cat_re_m1

# In[86]:


cat_re_m1 = cat(random_state=0,
                #              scale_pos_weight=2,
                #              class_weights={0:1,1:1.5},
                auto_class_weights='SqrtBalanced',
                iterations=1400,
                depth=7,
                learning_rate=0.1,
                custom_loss=['AUC', 'F1'],
                eval_metric='F1:use_weights=true',
                bagging_temperature=.5,
                verbose=False,
                use_best_model=True)
cat_re_m1.fit(x_train[select_var], y_train, eval_set=(
    july_df[select_var], july_df['inf_flag']), early_stopping_rounds=1295, plot=True)


# In[87]:


train_prediction = cat_re_m1.predict(june_train)
test_prediction = cat_re_m1.predict(june_test)

print("*****. TRAIN DATA: CatBoost MODEL spw=3 (Hardlines) ********")
print(sm.classification_report(train_prediction, june_y_train))


print("*****. VALIDATION DATA: CatBoost MODEL spw=3(Hardlines) ********")
print(sm.classification_report(test_prediction, june_y_test))

test_prediction = cat_re_m1.predict(july_df[select_var])


print("*****. July TEST DATA: CatBoost MODEL spw=3(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# #### cat_re_m2

# In[89]:


cat_re_m2 = cat(random_state=0,
                             scale_pos_weight=3,
#                              class_weights={0:1,1:1.5},
#                 auto_class_weights='Balanced',
                iterations=1400,
                depth=5,
                learning_rate=0.1,
                custom_loss=['AUC', 'F1'],
                eval_metric='F1:use_weights=true',
                bagging_temperature=.5,
                verbose=False,
                use_best_model=True)
cat_re_m2.fit(x_train[select_var], y_train, eval_set=(
    july_df[select_var], july_df['inf_flag']), early_stopping_rounds=1295, plot=True)

train_prediction = cat_re_m2.predict(june_train)
test_prediction = cat_re_m2.predict(june_test)

print("*****. TRAIN DATA: CatBoost MODEL spw=3 (Hardlines) ********")
print(sm.classification_report(train_prediction, june_y_train))


print("*****. VALIDATION DATA: CatBoost MODEL spw=3(Hardlines) ********")
print(sm.classification_report(test_prediction, june_y_test))

test_prediction = cat_re_m2.predict(july_df[select_var])


print("*****. July TEST DATA: CatBoost MODEL spw=3(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ### XGBOOST

# In[91]:


from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[856]:


xgb = XGBClassifier()
# param_distributions
param = {'n_estimators': [100, 150, 200, 300, 500],
         'booster': ['gbtree', 'dart'],
         'eta': [0.01, 0.05, 0.1, 0.2],
         'gamma': [1, 5, 10, 20, 50],
         'max_depth': [6, 10, 15, 20],
         'min_child_weight': [1, 3, 5, 10],
         'reg_lambda': [0, .1, .3, .5, 1, 2, 5, 10],
         'reg_alpha': [1, 10, 40, 60, 100, 120, 150, 180],
         'scale_pos_weight': [1, 3, 5, 7],
         'eval_metric': ['auc', 'aucpr', 'logloss']
         }


# In[863]:


rscv = RandomizedSearchCV(xgb,param,scoring='f1_weighted',cv=3,return_train_score=True)


# In[864]:


rscv_result = rscv.fit(june_train,june_y_train)


# In[865]:


rscv_result


# In[866]:


rscv_result.best_estimator_


# In[ ]:


rscv_result


# #### XGB_original Data

# In[103]:


# xgb = rscv_result.best_estimator_
xgb = XGBClassifier(base_score=0.5, booster='dart',
                    colsample_bynode=1,
                    colsample_bytree=1,
                    eta=0.2,
                    eval_metric='auc',
                    gamma=1,
                    importance_type='total_gain',
                    learning_rate=0.1,
                    max_delta_step=0,
                    max_depth=7,
                    min_child_weight=10,
                    n_estimators=100,
                    n_jobs=-1,
                    random_state=0,
                    reg_alpha=10,
                    reg_lambda=.1,
                    scale_pos_weight=2,
                    tree_method='exact')


# In[104]:


# june_train, june_test, june_y_train, june_y_test

xgb_1 = xgb.fit(june_train,june_y_train)


# In[105]:


train_prediction = xgb_1.predict(june_train)
test_prediction = xgb_1.predict(june_test)

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(june_y_train, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(june_y_test, test_prediction))


# In[106]:


test_prediction = xgb_1.predict(july_df[select_var])
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[ ]:





# ## Ensemble of Models

# In[109]:


m1 = m_o1 # lgbm.LGBMClassifier(n_estimators = 1000, max_depth=20,learning_rate=0.05,subsample=.5,reg_alpha=0.2,reg_lambda=0.2)
m2 = model_2 # lgbm.LGBMClassifier(n_estimators = 200, n_jobs = -1, max_depth=10,class_weight={0:2.5,1:1},num_leaves=50, importance_type='gain',boosting_type="dart" ) 
m3 = cat_m1 # Original_data
m4 = cat_m2 # Original_data
m5 = cat_m3 # Original_data
m6 = cat_re_m1
m7 = cat_re_m2


# In[110]:


# June original data prob_prediction M1

m1_june_prob = m1.predict_proba(june_test)[:, 1]
m2_june_prob = m2.predict_proba(june_test)[:, 1]
m3_june_prob = m3.predict_proba(june_test)[:, 1]
m4_june_prob = m4.predict_proba(june_test)[:, 1]
m5_june_prob = m5.predict_proba(june_test)[:, 1]
m6_june_prob = m6.predict_proba(june_test)[:, 1]
m7_june_prob = m7.predict_proba(june_test)[:, 1]

df_ens = pd.DataFrame({'m1_prob': m1_june_prob,
                       "m2_prob": m2_june_prob,
                       "m3_prob": m3_june_prob,
                       "m4_prob": m4_june_prob,
                       "m5_prob": m5_june_prob,
                       "m6_prob": m6_june_prob,
                       "m7_prob": m7_june_prob,
                       })


# In[112]:


# July

model_list = [m1, m2, m3, m4, m5, m6, m7]
prob_dict = {}

for x, y in enumerate(model_list):
    prob_dict['m' + str(x+1) +
              "_prob"] = y.predict_proba(july_df[select_var])[:, 1]


# ### lgr Model

# In[113]:


from sklearn.linear_model import LogisticRegression as lgr


# In[121]:


lgr_model = lgr(class_weight={0:1,1:2},max_iter=500)
log_m1 = lgr_model.fit(df_ens,june_y_test)


# In[122]:


lgr_prediction = log_m1.predict(df_ens)

print("*****. VALIDATION DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(lgr_prediction, june_y_test))


# In[123]:


jul_test_prediction = log_m1.predict(pd.DataFrame(prob_dict))

print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], jul_test_prediction))


# ### RF Model

# In[664]:


rfc_1 = rfc(n_estimators = 500,max_depth=3,n_jobs = -1)

rfc_m = rfc_1.fit(df_ens,june_y_test)


# In[665]:


rf_prediction = rfc_m.predict(df_ens)

print("*****. VALIDATION DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(rf_prediction, june_y_test))


# In[666]:



jul_test_prediction = rfc_m.predict(pd.DataFrame(prob_dict))

print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], jul_test_prediction))


# ### LGBM model

# In[124]:


lgbm_m = lgbm.LGBMClassifier(n_estimators = 1000, max_depth=3,n_jobs = -1,class_weight={0:.8,1:2})

lgbm_m = lgbm_m.fit(df_ens,june_y_test)


# In[125]:


lgbm_prediction = lgbm_m.predict(df_ens)

print("*****. VALIDATION DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(lgbm_prediction, june_y_test))


# In[126]:


jul_test_prediction = lgbm_m.predict(pd.DataFrame(prob_dict))

print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], jul_test_prediction))


# In[139]:


pd.DataFrame({'Vars':df_ens.columns,'importance':lgbm_m.feature_importances_}).set_index('Vars').sort_values(by='importance').plot(kind='barh',figsize=(10,5))
plt.show()


# ### CatBoost

# In[135]:


cat_en1 = cat(random_state=0,
             #              scale_pos_weight=5,
#              class_weights={0: 1, 1: 1.5},
             auto_class_weights='SqrtBalanced',
             iterations=1000,
             learning_rate=0.01,
             custom_loss=['PRAUC','F1'],
             eval_metric= 'PRAUC:use_weights=true',
             verbose=False,
             use_best_model=True)
cat_en1.fit(df_ens, june_y_test, eval_set=(
    pd.DataFrame(prob_dict), jul_test_prediction),
        early_stopping_rounds=600,
    plot=True)


# In[136]:


jul_test_prediction = cat_en1.predict(pd.DataFrame(prob_dict))

print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], jul_test_prediction))


# In[138]:


pd.DataFrame({'Vars':df_ens.columns,'importance':cat_en1.feature_importances_}).set_index('Vars').sort_values(by='importance').plot(kind='barh',figsize=(10,5))
plt.show()


# In[ ]:


'Logloss', 'CrossEntropy', 'CtrFactor', 'RMSE', 'LogCosh', 
'Lq', 'MAE', 'Quantile', 'Expectile', 'LogLinQuantile', 'MAPE', 
'Poisson', 'MSLE', 'MedianAbsoluteError', 'SMAPE', 'Huber', 'Tweedie', 
'Cox', 'RMSEWithUncertainty', 'MultiClass', 'MultiClassOneVsAll', 'PairLogit', 
'PairLogitPairwise', 'YetiRank', 'YetiRankPairwise', 'QueryRMSE', 'QuerySoftMax', 
'QueryCrossEntropy', 'StochasticFilter', 'LambdaMart', 'StochasticRank', 'PythonUserDefinedPerObject', 
'PythonUserDefinedMultiTarget', 'UserPerObjMetric', 'UserQuerywiseMetric', 'R2', 'NumErrors', 
'FairLoss', 'AUC', 'Accuracy', 'BalancedAccuracy', 'BalancedErrorRate', 'BrierScore', 'Precision', 
'Recall', 'F1', 'TotalF1', 'F', 'MCC', 'ZeroOneLoss', 'HammingLoss', 'HingeLoss', 'Kappa', 'WKappa', 
'LogLikelihoodOfPrediction', 'NormalizedGini', 'PRAUC', 'PairAccuracy', 'AverageGain', 'QueryAverage', 
'QueryAUC', 'PFound', 'PrecisionAt', 'RecallAt', 'MAP', 'NDCG', 'DCG', 'FilteredDCG', 'MRR', 'ERR', 
'SurvivalAft', 'MultiRMSE', 'MultiRMSEWithMissingValues', 'MultiLogloss', 'MultiCrossEntropy', 'Combination'.


# ### LOGITGAM

# In[690]:


import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines,CyclicCubicSplines
from statsmodels.gam.generalized_additive_model import LogitGam
# from statsmodels.gam.tests.test_penalized import df_autos


# In[732]:


def var_creat(df):
    df['allowance_c'] = pd.cut(df['allowance'],[0,3.5,7,10,13,16], include_lowest=True, labels=False)
    df['availableqty_c'] = pd.cut(df['availableqty'],[0,2,4,6,8,10,15], include_lowest=True, labels=False)
    df['bkrm_qt_c'] = pd.cut(june_df2['back_room_quantity'],[0,1,2,4,6,8,16,32,54,90,130,1167], include_lowest=True, labels=False)
    return df


# In[733]:


june_test = var_creat(june_test)
june_train = var_creat(june_train)
july_df = var_creat(july_df)


# In[735]:


june_test[['allowance_c','availableqty_c','bkrm_qt_c','received_hr']].describe()


# In[784]:


x_spline = june_test[['allowance_c','availableqty_c','bkrm_qt_c','received_hr']]
bs = BSplines(x_spline, df=[5,6,11,6], degree=[3, 4,4,3])


# In[785]:


transformed_val = bs.transform(june_test[['allowance_c','availableqty_c','bkrm_qt_c','received_hr']].values)


# In[786]:


transformed_df = pd.DataFrame(transformed_val,columns=bs.col_names)


# In[787]:


transformed_df.head()


# In[788]:


df_ens2 = df_ens.merge(transformed_df, right_index=True,left_index=True)


# In[814]:


lgbm_m = lgbm.LGBMClassifier(n_estimators = 1000, max_depth=10,n_jobs = -1,class_weight={0:.8,1:2})
lgbm_m = lgbm_m.fit(df_ens2,june_y_test)


# In[815]:


lgbm_prediction = lgbm_m.predict(df_ens2)

print("*****. VALIDATION DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(lgbm_prediction, june_y_test))


# In[ ]:


# transformed_val_jul = bs.transform(july_df[['allowance_c','availableqty_c','bkrm_qt_c','received_hr']].values)


# In[816]:


transformed_val_jul = bs.transform(july_df[['allowance_c','availableqty_c','bkrm_qt_c','received_hr']].values)
transformed_df_jul = pd.DataFrame(transformed_val_jul,columns=bs.col_names)
df_ens_jul = pd.DataFrame(prob_dict).merge(transformed_df_jul, right_index=True,left_index=True)


# In[817]:


jul_test_prediction = lgbm_m.predict(df_ens_jul)

print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], jul_test_prediction))


# ###  Data training and testing only on transformed data

# In[844]:


lgbm_m_tr = lgbm.LGBMClassifier(n_estimators = 2000,n_jobs = -1,class_weight={0:.8,1:2})
lgbm_m_tr = lgbm_m_tr.fit(june_train[['allowance','availableqty','back_room_quantity','received_hr']],june_y_train)


# In[845]:


jul_test_prediction = lgbm_m_tr.predict(july_df[['allowance','availableqty','back_room_quantity','received_hr']]) #transformed_val, june_y_test

print("*****. July TEST DATA: DEFAULT MODEL(Hardlines) ********")
print(sm.classification_report(july_df['inf_flag'], jul_test_prediction))


# In[ ]:





# ## Variable transformation (Scale + Power)

# In[ ]:





# In[443]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline



scaler = MinMaxScaler()
pt = PowerTransformer(method='yeo-johnson')
pipeline = Pipeline(steps=[('s', scaler), ('p', pt)])
# pipeline = Pipeline(steps=[('s', scaler)])

pipeline_ft = pipeline.fit(x_train)
trans_x_train = pipeline_ft.transform(x_train)
trans_x_test = pipeline_ft.transform(x_test)
trans_july_df = pipeline_ft.transform(july_df[select_var])


# In[445]:


# np.round(pipeline_ft['p'].lambdas_,2)


# In[446]:





# In[ ]:





# ### LGBM Model

# In[249]:


lgbm_1 = lgbm.LGBMClassifier(n_estimators = 250, n_jobs = -1, max_depth=20,class_weight={0:1,1:1.5},num_leaves=60, importance_type='gain'
                             ,boosting_type="dart" )

model_1 = lgbm_1.fit(trans_x_train,y_train)

train_prediction = model_1.predict(trans_x_train)
test_prediction = model_1.predict(trans_x_test)


# In[250]:


print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_1.predict(trans_july_df)
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ### RFC Model

# In[265]:


rfc_1 = rfc(n_estimators = 1000,n_jobs = -1)

rfc_model = rfc_1.fit(x_train,y_train)

train_prediction = rfc_model.predict(trans_x_train)
test_prediction = rfc_model.predict(trans_x_test)

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = rfc_model.predict(trans_july_df)
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ## cPCA transformation of data

# In[353]:


from contrastive import CPCA

c_pca = CPCA(standardize=True, verbose=True)


# In[360]:


# background = trans_x_train[np.where(y_train==0)]
background = x_train[y_train==0]
target = y_train


# In[325]:


# pca_X_trans  = c_pca.fit(trans_x_train[y_train],background,gui=True)


# In[88]:


c_pca.fit_transform(x_train[y_train],background)


# In[366]:


# pca_x_train = c_pca.fit_transform(trans_x_train,background,active_labels=y_train)
pca_x_train = c_pca.fit_transform(x_train,background,active_labels=y_train)

# print(pca_x_train.shape)

pca_x_test = c_pca.fit_transform(x_test,background,active_labels=y_train)

# print(pca_x_test.shape)

pca_x_july = c_pca.fit_transform(july_df[select_var],background,active_labels=y_train)


# In[367]:


np.shape(pca_x_train)


# ### LGBM after cPCA

# In[371]:


lgbm_trans = lgbm.LGBMClassifier(n_jobs = -1, importance_type='gain'
                             ,boosting_type="dart" )

model_trans = lgbm_trans.fit(pca_x_train[0],y_train)

train_prediction = model_trans.predict(pca_x_train[0])
test_prediction = model_trans.predict(pca_x_test[0])


# In[372]:


print("*****. cPCA TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. cPCA VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_trans.predict(pca_x_july[0])
print("*****. cPCA  TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[322]:


rfc_1 = rfc(n_estimators = 100,n_jobs = -1,max_depth=10)

rfc_model = rfc_1.fit(pca_x_train,y_train)

train_prediction = rfc_model.predict(pca_x_train)
test_prediction = rfc_model.predict(pca_x_test)

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = rfc_model.predict(pca_x_july)
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[ ]:





# ### Kernel PCA

# In[373]:


from sklearn.decomposition import KernelPCA as kpca


# In[377]:


kpca_m = kpca(n_components = 4, kernel='rbf',n_jobs=-1)


# In[378]:


transformer = kpca_m.fit(trans_x_train)


# ## Variable transformation (Scale)

# In[36]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

scaler = MinMaxScaler()
pt = PowerTransformer(method='yeo-johnson')
# pipeline = Pipeline(steps=[('s', scaler), ('p', pt)])
pipeline = Pipeline(steps=[('s', scaler)])

pipeline_ft = pipeline.fit(x_train)
trans_x_train = pipeline_ft.transform(x_train)
trans_x_test = pipeline_ft.transform(x_test)
trans_july_df = pipeline_ft.transform(july_df[select_var])


# ### Autoencoder

# In[32]:


import math
import pandas as pd
import tensorflow as tf
import kerastuner.tuners as kt
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError


# In[33]:


class AutoEncoders(Model):
    
    def __init__(self, output_units):
        super().__init__()
        self.encoder = Sequential(
            [
                Dense(output_units, activation="relu"),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(7, activation="relu")
            ]
        )
        self.decoder = Sequential(
            [
          Dense(32, activation="relu"),
          Dense(64, activation="relu"),
          Dense(output_units, activation="sigmoid")
            ]
        )
    
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# In[34]:


auto_encoder = AutoEncoders(trans_x_train.shape[1])

auto_encoder.compile(
    loss='mae',
    metrics=['mae'],
    optimizer='adam')

history = auto_encoder.fit(
    trans_x_train[y_train==0], 
    trans_x_train[y_train==0], 
    epochs=30, 
    batch_size=300, 
    validation_data=(trans_x_test[y_test==0], trans_x_test[y_test==0])
)


# In[35]:


encoder_layer = auto_encoder.get_layer('sequential')


# In[36]:


encoder_layer.output_shape


# In[37]:


reduced_df = pd.DataFrame(encoder_layer.predict(trans_x_train))
reduced_df = reduced_df.add_prefix('feature_')


# In[38]:


reduced_df


# In[39]:


reduced_df_test = pd.DataFrame(encoder_layer.predict(trans_x_test))
reduced_df_test = reduced_df_test.add_prefix('feature_')


# In[40]:


reduced_df_july = pd.DataFrame(encoder_layer.predict(trans_july_df))
reduced_df_july = reduced_df_july.add_prefix('feature_')


# ### Model Training LGBM

# In[471]:


lgbm_trans = lgbm.LGBMClassifier(n_estimators = 250,n_jobs = -1, importance_type='gain',class_weight={0:1,1:1.5}
                             ,boosting_type="dart" )

model_trans = lgbm_trans.fit(reduced_df,y_train)

train_prediction = model_trans.predict(reduced_df)
test_prediction = model_trans.predict(reduced_df_test)


# In[472]:


print("*****. Autoencoder TRAIN DATA (June): Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. Autoencoder VALIDATION DATA (June): Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_trans.predict(reduced_df_july)
print("*****. Autoencoder  TEST DATA (July): DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ### Model Training RF

# In[485]:


rfc_1 = rfc(n_estimators = 500, n_jobs = -1, max_depth=10,class_weight={0:.8,1:4},min_samples_leaf=2)

model_trans = rfc_1.fit(reduced_df,y_train)

train_prediction = model_trans.predict(reduced_df)


# In[486]:


test_prediction = model_trans.predict(reduced_df_test)


# In[487]:


print("*****. Autoencoder TRAIN DATA (June): Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. Autoencoder VALIDATION DATA (June): Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_trans.predict(reduced_df_july)
print("*****. Autoencoder  TEST DATA (July): DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[ ]:





# ### SVC

# In[41]:


from sklearn.svm import NuSVC, SVC


# In[495]:


svcm = SVC(kernel='rbf',class_weight='balanced')


# In[ ]:


model_svc = svcm.fit(reduced_df,y_train)

train_prediction = model_svc.predict(reduced_df)
test_prediction = model_svc.predict(reduced_df_test)


# In[ ]:


print("*****. SVC TRAIN DATA (June): Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


print("*****. SVC VALIDATION DATA (June): Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_svc.predict(reduced_df_july)
print("*****. SVC  TEST DATA (July): DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ## cPCA transformation

# ### Step1: Removing false positives

# In[80]:


lgbm_1 = lgbm.LGBMClassifier(n_estimators =300, n_jobs = -1, max_depth=10,class_weight={0:1.5,1:1},num_leaves=400,importance_type='gain'
                             ,boosting_type="dart" )
model_1 = lgbm_1.fit(x_train,y_train)


# In[81]:


train_prediction = model_1.predict(x_train)
test_prediction = model_1.predict(x_test)

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_train, train_prediction))


# In[82]:


x_train['y_train'] = y_train
x_train['y_predicted'] = train_prediction

x_train['sudo_y'] = np.where((x_train.y_train==0) & (x_train.y_predicted ==0),0,1)

x_train['false_positives'] = np.where((x_train.y_train==0) & (x_train.y_predicted ==1),1,0)

re_x_train = x_train[x_train.false_positives==0]

re_ytrain = re_x_train['y_train']
re_x_train = re_x_train.drop(columns=['y_predicted','sudo_y','y_train'])

re_x_train = re_x_train.drop(columns=['false_positives'])

x_train = x_train.drop(columns=['false_positives'])

x_train = x_train.drop(columns=['y_predicted','sudo_y','y_train'])


# In[83]:


x_train.shape


# In[84]:


re_x_train.shape


# In[146]:


# x_train.columns


# ### Minmax scaler

# In[92]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline



scaler = MinMaxScaler()
pt = PowerTransformer(method='yeo-johnson')
pipeline = Pipeline(steps=[('s', scaler), ('p', pt)])
# pipeline = Pipeline(steps=[('s', scaler)])

pipeline_ft = pipeline.fit(x_train)
trans_x_train = pipeline_ft.transform(x_train)
trans_x_retrain = pipeline_ft.transform(re_x_train)
trans_x_test = pipeline_ft.transform(x_test)
trans_july_df = pipeline_ft.transform(july_df[dep_var])


# ### cPCA transformation

# In[93]:


from contrastive import CPCA

c_pca = CPCA(standardize=False, verbose=True,n_components=2)


# In[94]:


background = trans_x_retrain[re_ytrain==0]
target = re_ytrain


# In[95]:


c_pca.fit_transform(trans_x_retrain, background, plot=True,active_labels=re_ytrain,gui=True,colors=['r','b','k','c'])


# In[ ]:


c_pca = CPCA(standardize=True, verbose=True,n_components=20)

# pca_x_train = c_pca.fit_transform(trans_x_train,background,active_labels=y_train)
pca_x_train = c_pca.fit_transform(trans_x_retrain,background,alpha_selection='manual', alpha_value=4.0,active_labels=re_ytrain)

# print(pca_x_train.shape)

pca_x_test = c_pca.fit_transform(trans_x_test,background,alpha_selection='manual', alpha_value=4.0,active_labels=re_ytrain)

# print(pca_x_test.shape)


# In[116]:


pca_x_july = c_pca.fit_transform(trans_july_df,background,active_labels=re_ytrain,alpha_selection='manual', alpha_value=100.73)


# In[97]:


type(pca_x_test)


# In[123]:


lgbm_trans = lgbm.LGBMClassifier(n_estimators = 300, n_jobs = -1,class_weight={0:1.5,1:1},num_leaves=300,importance_type='gain'
                             ,boosting_type="dart" )

model_trans = lgbm_trans.fit(pca_x_train,re_ytrain)


# In[124]:


train_prediction = model_trans.predict(pca_x_train)
test_prediction = model_trans.predict(pca_x_test)


# In[125]:


print("*****. cPCA TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(re_ytrain, train_prediction))


# In[126]:


print("*****. cPCA VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


# In[127]:


test_prediction = model_trans.predict(pca_x_july)
print("*****. cPCA  TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ### RFC

# In[135]:


lgbm_trans = rfc(n_estimators =200, n_jobs = -1, max_depth=10,class_weight={0:1.5,1:1},min_samples_leaf=1)

model_trans = lgbm_trans.fit(pca_x_train,re_ytrain)

train_prediction = model_trans.predict(pca_x_train)
test_prediction = model_trans.predict(pca_x_test)


# In[136]:


print("*****. cPCA TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(re_ytrain, train_prediction))


print("*****. cPCA VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_trans.predict(pca_x_july)
print("*****. cPCA  TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# ### LGBM Model: After false positves without transformation

# In[ ]:


lgbm_1 = lgbm.LGBMClassifier(n_estimators = 300, n_jobs = -1, max_depth=20,class_weight={0:1,1:1},num_leaves=300,importance_type='gain'
                             ,boosting_type="dart" )

model_1 = lgbm_1.fit(re_x_train,re_ytrain)

train_prediction = model_1.predict(re_x_train)
test_prediction = model_1.predict(x_test)


# In[138]:


print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(re_ytrain, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_1.predict(july_df[dep_var])
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[44]:


lgbm_1 = lgbm.LGBMClassifier(n_estimators = 500, n_jobs = -1, max_depth=20,class_weight={0:.8,1:2.5},num_leaves=300,importance_type='gain'
                             ,boosting_type="dart" )

model_1 = lgbm_1.fit(re_x_train,re_ytrain)

train_prediction = model_1.predict(re_x_train)
test_prediction = model_1.predict(x_test)

print("*****. TRAIN DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(re_ytrain, train_prediction))


print("*****. VALIDATION DATA: Model_1(Hardlines) ********" )
print(sm.classification_report(y_test, test_prediction))


test_prediction = model_1.predict(july_df[select_var])
print("*****. TEST DATA: DEFAULT MODEL(Hardlines) ********" )
print(sm.classification_report(july_df['inf_flag'], test_prediction))


# In[174]:


from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

precision_recall_curve(july_df['inf_flag'], test_prediction)


# In[177]:


precision, recall, overall = precision_recall_curve(july_df['inf_flag'], test_prediction)


# In[179]:


precision[1]


# ## Testing each variable performance

# In[199]:


lgbm_model = lgbm.LGBMClassifier(n_estimators = 200, n_jobs = -1, class_weight={0:.8,1:3.5},importance_type='gain'
                             ,boosting_type="dart" )
precision_score = []
recall_score = []
for var in select_var:
    model_fit = lgbm_model.fit(re_x_train[[var]],re_ytrain)
    test_prediction = model_fit.predict(x_test[[var]])
    precision, recall, overall = precision_recall_curve(y_test, test_prediction)
    precision_score.append(precision[1])
    recall_score.append(recall[1])


# In[200]:


precision_recall_df = pd.DataFrame({"Variables":select_var,"Precision":precision_score,"Recall":recall_score})
precision_recall_df = precision_recall_df.sort_values(by='Recall', ascending=False)
precision_recall_df


# In[ ]:





# In[ ]:


from IPython.display import display, Javascript
display(Javascript('IPython.notebook.save_checkpoint();'))


# In[ ]:





