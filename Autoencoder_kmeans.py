#!/usr/bin/env python
# coding: utf-8

# # Functions

# In[ ]:





# In[101]:


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
cred = json.load(open("credentails.json"))
gc.collect()
session=BigRed(username=cred['zid'], password=cred['pass'])


# In[2]:


# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import manifold


# In[4]:


# TensorFlow and Keras
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras


# In[102]:


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


# In[103]:


import warnings
warnings.filterwarnings('ignore')


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
    df['inf_rate'] = df['inf_q']/df['shp_req_q']
    for col in dep_var:
        df[col] = df[col].astype('float')
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
    rate_cols = ['availableqtywithoutthreshold', 'sales_1m', 'si_3_inf_rate', 'si_7_inf_rate', 'si_14_inf_rate', 'si_30_inf_rate',
                 'i_3_inf_rate', 'i_7_inf_rate', 'i_14_inf_rate', 'i_30_inf_rate',
                 'sdc_3_inf_rate', 'sdc_7_inf_rate', 'sdc_14_inf_rate',
                 '7d_inv_trnovr_rate', '7d_days_to_sell_inv', 'shp_oos_eoh_q',
                 '7d_min_shp_oos_eoh_q', 'allowance', 'release_qty']
    for x in rate_cols:
        dt[x] = np.where(dt[x] < 0, 0, dt[x])
        dt[x] = np.where(dt[x] < 0, 0, dt[x])

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


# ### Var Selection

# In[104]:


select_var = ['shp_req_q',
              'availableqty',
              'availableqtywithoutthreshold',
              'days_sold_year',
              'sales_year',
              'units_year',
              'days_sold_1m',
              'days_sold_1m_norm',
              'units_1m',
              'days_sold_7d',
              'units_7d',
              'sales_1m',
              'days_sold_3m',
              'units_3m',
              'sales_3m',
              'sales_3m_norm',
              'days_since_last_sale_both',
              'item_year_sales',
              'no_of_stores_sold',
              'store_year_sales',
              'store_year_units',
              'days_since_last_sale',
              'days_since_first_sale',
              'days_sold_norm',
              'avg_eoh_3d',
              'avg_eoh_2d',
              'sales_year_norm',
              'units_year_norm',
              'retl_a',
              'avg_eoh_7d',
              'avg_eoh_q',
              'avg_eoh_30d',
              'days_avail',
              'eoh_q_lt_3',
              'perc_eoh_lt_3',
              'ipi',
              'total_oh',
              'nbr_str',
              'avg_oh_network',
              'si_inf_rate',
              'weight',
              'back_room_quantity',
              'si_3_inf_rate',
              'si_7_inf_rate',
              'si_14_inf_rate',
              'si_30_inf_rate',
              'research_evnt_exists_3days',
              'any_evnt_exists_3days',
              'research_evnt_exists_7days',
              'any_evnt_exists_7days',
              'si_3_fulfill',
              'i_3_fulfill',
              's_3_fulfill',
              'sdc_3_fulfill',
              'dc_3_fulfill',
              'boh_q',
              'eoh_q',
              '7d_avg_boh_q',
              '7d_avg_eoh_q',
              '7d_boh_2_sls_rate',
              '7d_eoh_2_sls_rate',
              '7d_sell_thru_rate',
              '7d_inv_trnovr_rate',
              '7d_days_to_sell_inv',
              '7d_actual_oh_avg',
              'repln_q',
              '7d_repln_rate',
              '7d_dc_high_repln_rate',
              '7d_doh',
              'shp_oos_eoh_q',
              '7d_min_shp_oos_eoh_q',
              'sls_forcast_q',
              'threshold',
              'allowance',
              'release_qty',
              'received_hr']

# select_var2 = ['back_room_quantity',
# 'shp_req_q',
# 'availableqtywithoutthreshold',
# 'i_3_inf_rate',
# 'i_7_inf_rate',
# 'sdc_14_inf_rate',
# 'allowance',
# 'perc_eoh_lt_3',
# 'sdc_7_inf_rate',
# 'retl_a',
# 'eoh_q_lt_3',
# '7d_repln_rate',
# 'store_year_units',
# 'weight',
# '7d_dc_high_repln_rate',
# 'sales_3m',
# 'avg_eoh_q',
# 'dc_3_inf_rate',
# 'i_3_fulfill',
# 'i_14_inf_rate',
# 'release_qty',
# 'dc_14_inf_rate',
# 'availableqty',
# 'days_since_first_sale',
# '7d_eoh_2_sls_rate',
# 'si_3_fulfill',
# 'dc_30_inf_rate',
# 'nbr_str',
# '7d_actual_oh_avg',
# 'days_since_last_sale_both',
# 'avg_eoh_7d',
# 's_3_inf_rate',
# 'units_year_norm',
# 'received_hr',
# 'si_14_inf_rate',
# 'days_sold_3m',
# 'store_year_sales',
# 'no_of_stores_sold',
# 'boh_q',
# 'research_evnt_exists_3days',
# 'item_year_sales',
# 'sdc_3_fulfill',
# 'i_30_inf_rate',
# 's_7_inf_rate',
# 'sales_year',
# 'avg_eoh_30d',
# 'days_avail',
# 's_3_fulfill',
# 'total_oh',
# 'sales_year_norm',
# '7d_inv_trnovr_rate',
# 'avg_eoh_3d',
# 'any_evnt_exists_3days',
# 'dc_7_inf_rate',
# 'sales_1m',
# 's_30_inf_rate',
# 'eoh_q',
# 'units_1m',
# '7d_min_shp_oos_eoh_q',
# 'sls_forcast_q',
# 'avg_eoh_2d',
# 'days_sold_1m',
# 'avg_oh_network',
# 'days_sold_norm',
# 'days_sold_1m_norm',
# 'shp_oos_eoh_q',
# '7d_doh',
# 'sales_3m_norm',
# 'si_30_inf_rate',
# 'repln_q',
# '7d_days_to_sell_inv']


select_var2 = [
'back_room_quantity',
'allowance',
'i_30_inf_rate',
'availableqty',
'si_inf_rate',
'threshold',
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

var = dep_var + ['inf_flag','received_hr_c','inf_rate','inf_q']


# In[9]:





# In[105]:


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


# # Data Import & Treatment

# ## Data Import

# In[106]:


june_df = Preprep(session.query(query.format(6,10)))
june_df['received_hr_c'] = pd.cut(june_df['received_hr'],[0,6,10,17,20,24], include_lowest=True)
june_df['inf_rate'] = june_df['inf_q']/june_df['shp_req_q']
june_df = june_df[var]


# In[107]:


july_df = Preprep(session.query(query.format(7,10)))
july_df['received_hr_c'] = pd.cut(july_df['received_hr'],[0,6,10,17,20,24], include_lowest=True)
july_df['inf_rate'] = july_df['inf_q']/july_df['shp_req_q']
july_df = july_df[var]


# In[108]:


june_df.columns


# In[ ]:





# In[109]:


june_df['received_hr_c'] = pd.cut(june_df['received_hr'],[0,6,10,17,20,24], include_lowest=True, labels=False)


# In[110]:


july_df['received_hr_c'] = pd.cut(july_df['received_hr'],[0,6,10,17,20,24], include_lowest=True, labels=False)


# In[111]:


june_df['received_hr_c'].value_counts()


# ## Treatment

# In[112]:


summarydf = june_df.describe().T
summarydf[summarydf['min'] < 0].index

rate_cols = ['availableqtywithoutthreshold', 'sales_1m', 'si_3_inf_rate', 'si_7_inf_rate', 'si_14_inf_rate', 'si_30_inf_rate',
             'i_3_inf_rate', 'i_7_inf_rate', 'i_14_inf_rate', 'i_30_inf_rate',
             'sdc_3_inf_rate', 'sdc_7_inf_rate', 'sdc_14_inf_rate',
             '7d_inv_trnovr_rate', '7d_days_to_sell_inv', 'shp_oos_eoh_q',
             '7d_min_shp_oos_eoh_q', 'allowance', 'release_qty']

for x in rate_cols:
    june_df[x] = np.where(june_df[x] < 0,0,june_df[x])
    july_df[x] = np.where(july_df[x] < 0,0,july_df[x])
    


# In[113]:


## Outlier treatment
treated_df , iqr_list = outlier_treatment_df(june_df, depvar=dep_var)


# In[114]:


summary_original = summary(june_df)
summary_treated = summary(treated_df)

temp = summary_original[['coef_all']].merge(
    summary_treated[['coef_all']], right_index=True, left_index=True)

temp['coeff_dif'] = np.where(temp.coef_all_x > temp.coef_all_y, 0, 1)
var_for_treat = list(temp[temp.coeff_dif == 0].index)


treated_df , iqr_list = outlier_treatment_df(june_df, var_for_treat)

for col in var_for_treat:
        july_df['upper'] = iqr_list[col]
        july_df[col] = [y if x>y else x for x,y in zip(july_df[col],july_df['upper'])]
        july_df = july_df.drop(columns=['upper'])
        
        
for col in var_for_treat:
        june_df['upper'] = iqr_list[col]
        june_df[col] = [y if x>y else x for x,y in zip(june_df[col],june_df['upper'])]
        june_df = june_df.drop(columns=['upper'])
        


# # Clustering Kmeans

# ## Importing TF and keras packages

# In[338]:


clust_var = [
    'availableqtywithoutthreshold',
    'availableqty',
    'threshold',
    'back_room_quantity',
    'allowance',
    'i_30_inf_rate',
    'i_3_fulfill',
    'received_hr_c'

]


# In[339]:


inputs = {}
for name, column in june_df[clust_var].items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs


# In[340]:


numeric_inputs = {name: input for name, input in inputs.items() if input.dtype == tf.float32}


# In[341]:


x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(june_df[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
preprocessed_inputs = [all_numeric_inputs]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[342]:


preprocessed_inputs_cat = keras.layers.Concatenate()(preprocessed_inputs)
preprocessing_layer = tf.keras.Model(inputs, preprocessed_inputs_cat, name="ProcessData")


# In[343]:


# this saves an image of the model, see note regarding plot_model issues
tf.keras.utils.plot_model(model=preprocessing_layer, rankdir="LR", dpi=130, show_shapes=True, to_file="processing.png")


# In[218]:


# !pip install pydot


# In[344]:


items_features_dict = {name: np.array(value) for name, value in june_df[clust_var].items()}

# grab two samples
two_sample_dict = {name:values[1:3, ] for name, values in items_features_dict.items()}
two_sample_dict


# In[345]:


# apply the preprocessing layer
two_sample_fitted = preprocessing_layer(two_sample_dict)
two_sample_fitted


# ## Keras Model

# In[373]:


from keras.layers import LeakyReLU

# This is the size of our input data
full_dim = two_sample_fitted.shape.as_list()[1]

# these are the downsampling/upsampling dimensions
encoding_dim1 = 540
encoding_dim2 = 240
encoding_dim2a = 120
encoding_dim2b = 16
encoding_dim3 = 3
# we will use these 3 dimensions for clustering

# This is our encoder input
encoder_input_data = keras.Input(shape=(full_dim,))

# the encoded representation of the input
encoded_layer1 = keras.layers.Dense(encoding_dim1, activation='relu')(encoder_input_data)
encoded_layer2 = keras.layers.Dense(encoding_dim2, activation='relu')(encoded_layer1)
encoded_layer2a = keras.layers.Dense(encoding_dim2a, activation='relu')(encoded_layer2)
encoded_layer2b = keras.layers.Dense(encoding_dim2b, activation='relu')(encoded_layer2a)
# Note that encoded_layer3 is our 3 dimensional "clustered" layer, which we will later use for clustering
encoded_layer3 = keras.layers.Dense(encoding_dim3, activation='relu', name="ClusteringLayer")(encoded_layer2b)

encoder_model = keras.Model(encoder_input_data, encoded_layer3)

# the reconstruction of the input
decoded_layer3 = keras.layers.Dense(encoding_dim1, activation='relu')(encoded_layer3)
decoded_layer2a = keras.layers.Dense(encoding_dim2a, activation='relu')(decoded_layer3)
decoded_layer2 = keras.layers.Dense(encoding_dim2b, activation='relu')(decoded_layer2a)
decoded_layer1 = keras.layers.Dense(full_dim, activation='sigmoid')(decoded_layer2)

# This model maps an input to its autoencoder reconstruction
autoencoder_model = keras.Model(encoder_input_data, outputs=decoded_layer1, name="Encoder")

# compile the model
autoencoder_model.compile(optimizer="ADAM", loss=tf.keras.losses.mean_squared_logarithmic_error)
# tf.keras.utils.plot_model(model=autoencoder_model, rankdir="LR", dpi=130, show_shapes=True, to_file="autoencoder.png") #RMSProp


# In[374]:


encoder_model.summary()


# In[441]:


autoencoder_model.summary()


# ### Model deployment

# In[376]:


# process the inputs
p_items = preprocessing_layer(items_features_dict)
p_labels = june_df['inf_flag']

# split into training and testing sets (80/20 split)
train_data, test_data, train_labels, test_labels = train_test_split(p_items.numpy(), p_labels, train_size=0.7, random_state=5)

# fit the model using the training data
history = autoencoder_model.fit(train_data, train_data, epochs=100, batch_size=560, shuffle=True, validation_data=(test_data, test_data))


# In[127]:





# In[377]:


# encoder_model.save("encoder_model.h5")

encoder_model.save("encoder_save", overwrite=True)


# In[378]:


from IPython.display import display, Javascript
display(Javascript('IPython.notebook.save_checkpoint();'))


# In[379]:


encoder_save = tf.keras.models.load_model("encoder_save")


# In[380]:


encoder_save.summary()


# In[408]:


encoded_items = encoder_save(p_items)


# In[ ]:





# ## Cluster elbow chart

# In[382]:


# encoded_items = encoder_model(p_items)

# choose number of clusters K:
Sum_of_squared_distances = []
K = range(2,6)
for k in K:
    km = KMeans(init='k-means++', n_clusters=k, n_init=10)
    km.fit(encoded_items)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.vlines(ymin=0, ymax=150000, x=4, colors='red')
plt.text(x=8.2, y=130000, s="optimal K=8")
plt.xlabel('Number of Clusters K')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal K')
plt.show()


# In[442]:


# encoded_items = encoder_model(p_items)

# choose number of clusters K:
Sum_of_squared_distances = []
K = range(2,10)
for k in K:
    km = KMeans(init='k-means++', n_clusters=k, n_init=10)
    km.fit(encoded_items)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.vlines(ymin=0, ymax=150000, x=4, colors='red')
plt.text(x=8.2, y=130000, s="optimal K=4")
plt.xlabel('Number of Clusters K')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal K')
plt.show()


# In[384]:


from IPython.display import display, Javascript
display(Javascript('IPython.notebook.save_checkpoint();'))


# ## Cluster pred June data

# In[409]:


km = KMeans(init='k-means++', n_clusters=4, n_init=10)
km.fit(encoded_items)
predicted_clusters = km.predict(encoded_items)


# In[ ]:





# In[410]:


predicted_clusters.shape


# In[411]:


june_df.shape


# In[412]:


june_df['pred_cluster'] = predicted_clusters

june_df.groupby('pred_cluster')['inf_flag','allowance','availableqty','availableqtywithoutthreshold','back_room_quantity','received_hr'].describe().T


# In[413]:


import matplotlib.pyplot as plt
import seaborn as sns

clust_var = [
    'availableqtywithoutthreshold',
    'availableqty',
    'threshold',
    'back_room_quantity',
    'allowance',
    'i_30_inf_rate',
    'i_3_fulfill',
    'received_hr_c'

]

# plt.hist(june_df2[june_df2.inf_flag==1]['availableqty'], bins=12,color='green')
sns.boxplot(y=june_df['availableqty'], x=june_df['pred_cluster'])
plt.title('availableqty')
plt.show()

sns.boxplot(y=june_df['allowance'], x=june_df['pred_cluster'])
plt.title('allowance')
plt.show()

sns.boxplot(y=june_df['back_room_quantity'][june_df.back_room_quantity <= 20], x=june_df['pred_cluster'])
plt.title('back_room_quantity')
plt.show()


sns.boxplot(y=june_df['availableqtywithoutthreshold'], x=june_df['pred_cluster'])
plt.title('availableqtywithoutthreshold')
plt.show()



sns.boxplot(y=june_df['threshold'], x=june_df['pred_cluster'])
plt.title('threshold')
plt.show()


# In[414]:


def iqr_val(col):
    '''
    returns the upper limit for outliers
    '''
    q1,q3 = np.quantile(col,[.25,.75])
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


# In[416]:


def outlier_treatment(main_df,depvar,groupvar=None,iqr_range=1.5):
    '''
    input : main_df, depvar, group(default=None),iqr_range=1.5
    return's : df, iqr_list
    '''
    def iqr_val(col):
        '''
        returns the upper limit for outliers'''
        q1,q3 = np.quantile(col,[.25,.75])
        iqr = q3 - q1
        return q3 + iqr_range*iqr

    
    if groupvar==None:
        df = main_df.copy()
        iqr_list = {}
        for col in depvar:
            iqr_list[col] = iqr_val(df[col])
        for col in depvar:
            df['upper'] = iqr_list[col]
            df[col] = [y if x > y else x for x, y in zip(df[col], df['upper'])]
            df = df.drop(columns=['upper'])
        return df, iqr_list
    else:
        df = main_df.copy()
        iqr_group = {}
        for col in depvar:
            iqr_group[col] = df.groupby(groupvar)[col].apply(lambda x: iqr_val(x)).to_dict()
        for col in depvar:
            df['upper'] = df[groupvar].map(iqr_group[col])
            df[col] = [y if (x > y) else x for x,y in zip(df[col],df['upper'])]
            df = df.drop(columns=['upper'])
        return df, iqr_group 
            
            
           


# ## Outlier treatment by clusters

# In[417]:


## Test
# june_df[['allowance','availableqty','back_room_quantity','received_hr','pred_cluster']].groupby('pred_cluster').apply(lambda x: outlier_treatment(x))


# In[418]:


result_df, iqr_group = outlier_treatment(june_df,select_var2,'pred_cluster')


# In[419]:


iqr_group


# In[ ]:





# In[420]:


# plt.hist(june_df2[june_df2.inf_flag==1]['availableqty'], bins=12,color='green')
sns.boxplot(y=result_df['availableqty'], x=result_df['pred_cluster'])
plt.title('availableqty')
plt.show()

sns.boxplot(y=result_df['allowance'], x=result_df['pred_cluster'])
plt.title('allowance')
plt.show()

sns.boxplot(y=result_df['back_room_quantity'], x=result_df['pred_cluster'])
plt.title('back_room_quantity')
plt.show()


sns.boxplot(y=result_df['received_hr'], x=result_df['pred_cluster'])
plt.title('received_hr')
plt.show()


# ## Cluster pred July Data

# In[425]:


items_features_dict = {name: np.array(value) for name, value in july_df[clust_var].items()}

p_items_july = preprocessing_layer(items_features_dict)

encoded_items = encoder_model(p_items_july)


# In[426]:


predicted_clusters = km.predict(encoded_items)


# In[427]:


july_df['pred_cluster'] = predicted_clusters


# In[428]:


# plt.hist(june_df2[june_df2.inf_flag==1]['availableqty'], bins=12,color='green')
sns.boxplot(y=july_df['availableqty'], x=july_df['pred_cluster'])
plt.title('availableqty')
plt.show()

sns.boxplot(y=july_df['allowance'], x=july_df['pred_cluster'])
plt.title('allowance')
plt.show()

sns.boxplot(y=july_df['back_room_quantity'][july_df.back_room_quantity <= 20], x=july_df['pred_cluster'])
plt.title('back_room_quantity')
plt.show()


sns.boxplot(y=july_df['received_hr'], x=july_df['pred_cluster'])
plt.title('received_hr')
plt.show()


# In[396]:


# iqr_group


# In[429]:


for x in iqr_group.keys():
    july_df['upper'] = july_df['pred_cluster'].map(iqr_group[x])
    july_df[x] = [y if (x > y) else x for x,y in zip(july_df[x],july_df['upper'])]
    july_df = july_df.drop(columns='upper')


# In[430]:


# plt.hist(june_df2[june_df2.inf_flag==1]['availableqty'], bins=12,color='green')
sns.boxplot(y=july_df['availableqty'], x=july_df['pred_cluster'])
plt.title('availableqty')
plt.show()

sns.boxplot(y=july_df['allowance'], x=july_df['pred_cluster'])
plt.title('allowance')
plt.show()

sns.boxplot(y=july_df['back_room_quantity'][july_df.back_room_quantity <= 20], x=july_df['pred_cluster'])
plt.title('back_room_quantity')
plt.show()


sns.boxplot(y=july_df['received_hr'], x=july_df['pred_cluster'])
plt.title('received_hr')
plt.show()


# # Modeling by Clusters(Kmeans)

# ## Catboost

# In[431]:


def model_train(model_df,depvar,target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number
    
    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(df[depvar],df[target], train_size=0.7)
    cat_m3 = cat(random_state=0,
             scale_pos_weight=3,
#              auto_class_weights='SqrtBalanced',
             iterations=1000,
             depth=10,
             learning_rate=0.1,
             custom_loss=['AUC', 'F1'],
             eval_metric= 'PRAUC', #'F1:use_weights=true',
             verbose=False,
             bagging_temperature=.5,
             use_best_model=True)
    cat_m3.fit(x_train, y_train, eval_set=(
        x_test, y_test),
        early_stopping_rounds=50,
        plot=True)
    
    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(f"*****. TRAIN DATA: Cluster {cluster_num} CatBoost MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))
    
    print(f"*****. VALIDATION DATA: Cluster {cluster_num} CatBoost MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    
    return cat_m3


# In[432]:


cluster_0 =  model_train(june_df,select_var2,'inf_flag',0)


# In[433]:


test_prediction = cluster_0.predict(july_df[july_df.pred_cluster==0][select_var2])
print("***** Cluster Number:0 , July TEST DATA: CatBoost MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==0]['inf_flag'], test_prediction))


# In[434]:



def model_train(model_df,depvar,target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number
    
    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(df[depvar],df[target], train_size=0.7)
    cat_m3 = cat(random_state=0,
#              scale_pos_weight=3,
             auto_class_weights='SqrtBalanced',
             iterations=1000,
#              depth=10,
             learning_rate=0.1,
             custom_loss=['PRAUC', 'F1'],
             eval_metric= 'AUC', #'F1:use_weights=true',
             verbose=False,
             use_best_model=True)
    cat_m3.fit(x_train, y_train, eval_set=(
        x_test, y_test),
        early_stopping_rounds=50,
        plot=True)
    
    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(f"*****. TRAIN DATA: Cluster {cluster_num} CatBoost MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))
    
    print(f"*****. VALIDATION DATA: Cluster {cluster_num} CatBoost MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    
    return cat_m3




clus_num = 1
cluster_1 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_1.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: CatBoost MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[435]:


clus_num = 2

cluster_2 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_2.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: CatBoost MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[272]:


def model_train(model_df,depvar,target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number
    
    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(df[depvar],df[target], train_size=0.7)
    cat_m3 = cat(random_state=0,
#              scale_pos_weight=3,
             auto_class_weights='SqrtBalanced',
             iterations=1000,
#              depth=10,
             learning_rate=0.1,
             custom_loss=['PRAUC', 'F1'],
             eval_metric='F1:use_weights=true',
             verbose=False,
             use_best_model=True)
    cat_m3.fit(x_train, y_train, eval_set=(
        x_test, y_test),
        early_stopping_rounds=500,
        plot=True)
    
    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(f"*****. TRAIN DATA: Cluster {cluster_num} CatBoost MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))
    
    print(f"*****. VALIDATION DATA: Cluster {cluster_num} CatBoost MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    
    return cat_m3




clus_num = 3

cluster_3 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_3.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: CatBoost MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# ## LGBM

# In[436]:


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_estimators=300, n_jobs=-1, max_depth=10, class_weight={
        0:1, 1: 1.5}, num_leaves=60, importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=50)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3


# In[ ]:





# In[437]:


clus_num = 0

cluster_0 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_2.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[438]:


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_estimators=500, n_jobs=-1, max_depth=12, class_weight='balanced', 
                                num_leaves=60, importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=50)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3

clus_num = 1
cluster_1 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_1.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# 

# In[439]:


clus_num = 2

cluster_2 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_2.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[440]:


july_df['pred_cluster'].value_counts()


# # DO NOT SEE AFFTER THAT

# In[85]:


'''
lgbm.LGBMClassifier(n_estimators=1000, n_jobs=-1, max_depth=10, 
class_weight={ 0:1, 1: 1.5}, num_leaves=60, importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=50)
'''


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    
    
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_jobs=-1, max_depth=10,                                 class_weight={ 0:1, 1: 2}, num_leaves=60, importance_type='gain',                                boosting_type="dart", reg_alpha=.3, reg_lambda=50)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()
    print(model)

    return cat_m3


# In[84]:





# In[ ]:





# In[82]:


clus_num = 2

cluster_2 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_2.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[ ]:





# In[59]:


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number
     max_depth=10

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_estimators=200, n_jobs=-1,class_weight={0:.5,1:3},                                importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=50)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3


# In[551]:


clus_num = 2

cluster_2 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_2.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# ## previous run 

# In[426]:


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_estimators=1000, n_jobs=-1, max_depth=15, class_weight='balanced', num_leaves=60, importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=.5)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3


# In[427]:


clus_num = 1
print(f"Cluster Number:{clus_num}")

cluster_1 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_1.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[428]:


clus_num = 0
print(f"Cluster Number:{clus_num}")

cluster_0 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_0.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[429]:


clus_num = 2
print(f"Cluster Number:{clus_num}")

cluster_2 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_2.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# ## Model-clust_var

# In[468]:


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_estimators=1000, n_jobs=-1, max_depth=15, class_weight='balanced', num_leaves=60, importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=.5)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3


# In[469]:


clus_num = 0
print(f"Cluster Number:{clus_num}")

cluster_0 =  model_train(june_df,clust_var,'inf_flag',clus_num)

test_prediction = cluster_0.predict(july_df[july_df.pred_cluster==clus_num][clust_var])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[470]:


clus_num = 1
print(f"Cluster Number:{clus_num}")

cluster_1 =  model_train(june_df,clust_var,'inf_flag',clus_num)

test_prediction = cluster_1.predict(july_df[july_df.pred_cluster==clus_num][clust_var])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[471]:


clus_num = 2
print(f"Cluster Number:{clus_num}")

cluster_2 =  model_train(june_df,clust_var,'inf_flag',clus_num)

test_prediction = cluster_2.predict(july_df[july_df.pred_cluster==clus_num][clust_var])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# In[223]:


from sklearn.svm import SVC

def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.pred_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = SVC(C=.5, class_weight='balanced')
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} CatBoost MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} CatBoost MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3


# In[224]:


clus_num = 0
print(f"Cluster Number:{clus_num}")

cluster_1 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_1.predict(july_df[july_df.pred_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: SVM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.pred_cluster==clus_num]['inf_flag'], test_prediction))


# # Clustering SOM

# In[204]:


from sklearn_som.som import SOM


# In[205]:


som = SOM(m=2, n=2, dim=3,random_state=1,lr=2,max_iter=5000)


# In[206]:


som.shape


# In[207]:


som.cluster_centers_


# In[208]:


encoded_items.shape


# In[209]:


# encoded_items = encoder_model(p_items)
som.fit(encoded_items)


# ### Cluster prediction

# In[210]:


prediction = som.predict(encoded_items)


# In[211]:


june_df['som_cluster'] = prediction


# In[ ]:


items_features_dict = {name: np.array(value) for name, value in july_df[clust_var].items()}

p_items_july = preprocessing_layer(items_features_dict)

encoded_items = encoder_model(p_items_july)

prediction_july = som.predict(p_items_july)
july_df['som_cluster'] = prediction_july


# In[ ]:


june_df.groupby('som_cluster')['inf_flag','allowance','availableqty','back_room_quantity','received_hr_c'].describe().T


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# plt.hist(june_df2[june_df2.inf_flag==1]['availableqty'], bins=12,color='green')
sns.boxplot(y=june_df['availableqty'], x=june_df['som_cluster'])
plt.title('availableqty')
plt.show()

sns.boxplot(y=june_df['allowance'], x=june_df['som_cluster'])
plt.title('allowance')
plt.show()

sns.boxplot(y=june_df['back_room_quantity'][june_df.back_room_quantity <= 20], x=june_df['som_cluster'])
plt.title('back_room_quantity')
plt.show()


sns.boxplot(y=june_df['received_hr_c'], x=june_df['som_cluster'])
plt.title('received_hr')
plt.show()


# ## SOM modeling

# In[ ]:


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.som_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_estimators=2000, n_jobs=-1, max_depth=10, class_weight={
        0:1.5, 1: 1}, num_leaves=60, importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=50)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3


# #### pred_cluster1

# In[ ]:


clus_num = 1

cluster_1 =  model_train(june_df,select_var2,'inf_flag',clus_num)

test_prediction = cluster_1.predict(july_df[july_df.som_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")

print(sm.classification_report(july_df[july_df.som_cluster==clus_num]['inf_flag'], test_prediction))


# #### pred_cluster_2

# In[ ]:


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.som_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_estimators=2000, n_jobs=-1, max_depth=20, class_weight={
        0:1.5, 1: 1}, num_leaves=70, importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=50)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3


clus_num = 2
cluster_2 =  model_train(june_df,select_var2,'inf_flag',clus_num)
test_prediction = cluster_2.predict(july_df[july_df.som_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.som_cluster==clus_num]['inf_flag'], test_prediction))


# #### pred_cluster_3

# In[ ]:


def model_train(model_df, depvar, target, cluster_num):
    '''
    model_df = main data_frame
    depvar = dependendent variable list
    target = target variable
    cluster_num = cluster number

    '''
    from catboost import CatBoostClassifier as cat
    df = model_df[model_df.som_cluster == cluster_num]
    x_train, x_test, y_train, y_test = train_test_split(
        df[depvar], df[target], train_size=0.7)
    model = lgbm.LGBMClassifier(n_estimators=1000, n_jobs=-1, max_depth=10, class_weight={
        0:1.5, 1: 1}, num_leaves=70, importance_type='gain', boosting_type="dart", reg_alpha=.3, reg_lambda=50)
    cat_m3 = model.fit(x_train, y_train)

    train_prediction = cat_m3.predict(x_train)
    test_prediction = cat_m3.predict(x_test)
    print(
        f"*****. TRAIN DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(train_prediction, y_train))

    print(
        f"*****. VALIDATION DATA: Cluster {cluster_num} LGBM MODEL (Hardlines) ********")
    print(sm.classification_report(test_prediction, y_test))
    gc.collect()

    return cat_m3


clus_num = 3
cluster_3 =  model_train(june_df,select_var2,'inf_flag',clus_num)
test_prediction = cluster_3.predict(july_df[july_df.som_cluster==clus_num][select_var2])
print(f"***** Cluster Number:{clus_num} , July TEST DATA: LGBM MODEL (Hardlines) ********")
print(sm.classification_report(july_df[july_df.som_cluster==clus_num]['inf_flag'], test_prediction))


# # Graphs

# In[560]:


sns.boxenplot(y=june_df['received_hr'],x=june_df['inf_flag'])


# In[41]:


plt.figure(figsize=(10,8))
sns.boxenplot(x=june_df['received_hr_c'],hue=june_df['inf_flag'],y=june_df['inf_q'])
plt.show()


# In[67]:


np.log(june_df['inf_rate'])


# In[168]:


sns.boxplot(y=june_df['received_hr_c'], hue=june_df['inf_flag'])
plt.title('received_hr')
plt.show()


# In[165]:


sns.boxplot(y=june_df['received_hr_c'][june_df.inf_q>0],x=june_df['inf_q'][june_df.inf_q>0])
plt.title('Includes only INF=1')
plt.show()


# In[109]:


plt_df = june_df[['received_hr_c','inf_q','inf_flag']][june_df.inf_q>0].groupby('received_hr_c')['inf_flag'].sum()/june_df['inf_flag'].sum()*100


# In[120]:


plt_df


# In[122]:


plt_df.plot(kind='barh')
plt.title('Percentage(%) of Total Inf by received_hr')
plt.show()


(june_df[['received_hr_c','inf_q','inf_flag','inf_rate']].groupby('received_hr_c')['inf_flag'].agg(['mean','std'])['mean']*100).plot()
plt.title('Inf rate by received_hr')
plt.show()


# In[117]:


perc = [.25,.5,.75,.9,.99]
(june_df[['received_hr_c','inf_q','inf_flag','inf_rate']].groupby('received_hr_c')['inf_flag'].agg(['mean','std'])['mean']*100).plot()
plt.title('Inf rate by received_hr')
plt.show()


# In[184]:


june_df['received_hr_c'] = pd.cut(june_df['received_hr'],[0,6,10,17,20,24], include_lowest=True)
result_df = june_df.groupby('received_hr_c')['inf_flag'].agg(['sum','count','mean'])
result_df.columns = ['Inf','Total','Inf_rate']
result_df['Inf_rate'].plot(kind='barh')
plt.title("Inf_rate by 'Received_hour category' ")
plt.show()
result_df['Inf_rate'] = result_df['Inf_rate']*100
result_df


# In[118]:


import scipy.stats as scst


# In[182]:


result = scst.kruskal(
june_df['received_hr_c'][june_df.inf_rate ==1],
june_df['received_hr_c'][june_df.inf_rate ==0])

print("Kruskal-Wallis H test for 'Received Hours':")
print(f"kw_statistic :{round(result[0],2)},kw_pvalue={round(result[1],2)}")


# ***Test statistics for Kruskal-Wallis H test shows that there is significant difference 'Received Hours' by inf category***



june_df['received_hr_c'] = june_df['received_hr_c'].astype(str)




