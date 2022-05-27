from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer(n_quantiles=50,output_distribution='normal')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer(n_quantiles=50,output_distribution='normal')

var = [
 'Attribute16',
 'avg_pt_forecast',
 'avg_store_sales',
 'median_pt_forecast',
 'missed_wtdr_14',
 'missed_wtdr_sum_fcst',
 'pop_boh_aftr_dlvry_14',
 'pop_missed_wtdr_iip_14',
 'pop_onorder_14',
 'pop_sch_receipt_14',
 'prj_rc_boh',
 'prj_store_boh',
 'projected_wtdr_minus_unfill_14',
 'sum_fcst_14',
 'store_boh_vs_avgfcst_r']

scl_model = scaler.fit(train_2m[var])

train_2m_scl = scl_model.transform(train_2m[var])
test_2m_scl = scl_model.transform(test_2m[var])

train_2m_scl_1 = scl_model.transform(train_2m[train_2m.target==1][var])
train_2m_scl_0 = scl_model.transform(train_2m[train_2m.target==0][var])



test_2m_scl_1 = scl_model.transform(test_2m[test_2m.target==1][var])
test_2m_scl_0 = scl_model.transform(test_2m[test_2m.target==0][var])

test_june_scl = scl_model.transform(test_june[var])

test_june_scl_1 = scl_model.transform(test_june[test_june.target==1][var])
test_june_scl_0 = scl_model.transform(test_june[test_june.target==0][var])

### Ploting graphs

from pylab import rcParams
%matplotlib inline
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 25, 20

pd.DataFrame(test_2m_scl_1, columns=var).hist()
plt.title('test_2m_scl_1')
plt.show()

pd.DataFrame(test_june_scl_1, columns=var).hist()
plt.title('test_june_scl_1')
plt.show()



## Using Auotencoder for anomaly detection

from scipy import stats
import tensorflow as tf


from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras import models, metrics
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from pylab import rcParams

from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.layers import BatchNormalization

from keras.metrics import PrecisionAtRecall, SensitivityAtSpecificity



### Anomaly training on target_1

input_dim = train_2m_scl_1.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(input_dim,activation='tanh',activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(input_dim / 2), activation="relu")(encoder)
encoder = Dense(int(input_dim / 4), activation="relu")(encoder)
decoder = Dense(int(input_dim / 2), activation="tanh")(encoder)
decoder = Dense(input_dim, activation="relu")(decoder)

autoencoder = models.Model(inputs=input_layer,outputs=decoder)

autoencoder.summary()

nb_epoch = 500
batch_size = 30
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics='accuracy')
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(train_2m_scl_1, train_2m_scl_1,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test_2m_scl_1, test_2m_scl_1),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');

predictions = autoencoder.predict(train_2m_scl)

mse = np.mean(np.power(train_2m_scl - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': train_2m['target']})

error_df.describe()

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve',linestyle='--')
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision & Recall')
plt.show()


threshold = 15

## And see how well we’re dividing the two types of transactions:

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "OOS" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();

LABELS = ["Normal", "OOS"]
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

train_2m['anom_pred_1'] = y_pred

#### Test anomaly predict

predictions = autoencoder.predict(test_2m_scl)

mse = np.mean(np.power(test_2m_scl - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': test_2m['target']})
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
test_2m['anom_pred_1'] = y_pred

#### Test_june anomaly predict

predictions = autoencoder.predict(test_june_scl)

mse = np.mean(np.power(test_june_scl - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': test_june['target']})
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
test_june['anom_pred_1'] = y_pred

### Anomaly training on target_0

input_dim = train_2m_scl_0.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(input_dim,activation='tanh',activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(input_dim / 2), activation="relu")(encoder)
encoder = Dense(int(input_dim / 4), activation="relu")(encoder)
decoder = Dense(int(input_dim / 2), activation="tanh")(encoder)
decoder = Dense(input_dim, activation="relu")(decoder)
autoencoder = models.Model(inputs=input_layer,outputs=decoder)
autoencoder.summary()

nb_epoch = 100
batch_size = 1000
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics='accuracy')
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(train_2m_scl_0, train_2m_scl_0,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test_2m_scl_0, test_2m_scl_0),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');

predictions = autoencoder.predict(train_2m_scl)

mse = np.mean(np.power(train_2m_scl - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': train_2m['target']})

error_df.describe()

fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();

threshold = 16.3

## And see how well we’re dividing the two types of transactions:

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "OOS" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();

predictions = autoencoder.predict(train_2m_scl)

mse = np.mean(np.power(train_2m_scl - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': train_2m['target']})
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
train_2m['anom_pred_0'] = y_pred

predictions = autoencoder.predict(test_2m_scl)

mse = np.mean(np.power(test_2m_scl - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': test_2m['target']})
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
test_2m['anom_pred_0'] = y_pred

predictions = autoencoder.predict(test_june_scl)

mse = np.mean(np.power(test_june_scl - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': test_june['target']})
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
test_june['anom_pred_0'] = y_pred

#### Sudo Target creation on autoencoder anomaly target

train_2m['sudo_target'] = np.where((train_2m['anom_pred_1']==1) | (train_2m['anom_pred_0']==-1),1,0)
test_2m['sudo_target'] = np.where((test_2m['anom_pred_1']==1) | (test_2m['anom_pred_0']==-1),1,0)
test_june['sudo_target'] = np.where((test_june['anom_pred_1']==1) | (test_june['anom_pred_0']==-1),1,0)

### Outlier Detection by sudo_target

var = [
 'avg_pt_forecast',
 'avg_store_sales',
 'median_pt_forecast',
 'missed_wtdr_14',
 'missed_wtdr_sum_fcst',
 'pop_boh_aftr_dlvry_14',
 'pop_missed_wtdr_iip_14',
 'pop_onorder_14',
 'pop_sch_receipt_14',
 'prj_rc_boh',
 'prj_store_boh',
 'projected_wtdr_minus_unfill_14',
 'sum_fcst_14',
 'store_boh_vs_avgfcst_r']


iqr_group = {}
for col in var:
    iqr_group[col] = train_2m.groupby('sudo_target')[col].apply(lambda x: outlier_treatment(x)).to_dict()



for col in var:
    train_2m['upper'] = train_2m['sudo_target'].map(iqr_group[col])
    train_2m['n2_'+col] = [y if (x > y) else x for x,y in zip(train_2m[col],train_2m['upper'])]

for col in var:
    test_2m['upper'] = test_2m['sudo_target'].map(iqr_group[col])
    test_2m['n2_'+col] = [y if (x > y) else x for x,y in zip(test_2m[col],test_2m['upper'])]
    
### After treatment

svar = [ 
#     'Attribute16',
#        'n2_missed_wtdr_sum_fcst',
 'n2_avg_pt_forecast',
 'n2_avg_store_sales',
 'n2_median_pt_forecast',
 #'n2_missed_wtdr_14',
 #'n2_pop_boh_aftr_dlvry_14',
 #'n2_pop_missed_wtdr_iip_14',
 'n2_pop_onorder_14',
 'n2_pop_sch_receipt_14',
 'n2_prj_rc_boh',
 'n2_prj_store_boh',
 'n2_projected_wtdr_minus_unfill_14',
 'n2_sum_fcst_14'
#  'n2_store_boh_vs_avgfcst_r'
]

default_model = RandomForestClassifier(n_estimators=30,criterion='gini',warm_start=True,n_jobs=-1,
                                       oob_score=True)

model_rf = default_model.fit(train_2m[svar],train_2m['target'])

test_predict = model_rf.predict(test_2m[svar])

predict_def = pd.DataFrame({'target':test_2m['target'],
                              'predict':test_predict})


report_def_2m = classification_report(predict_def['target'], predict_def['predict'])


print("Variables used:",'\n',svar,'\n')

print('\n')
print(default_model)
print('\n')
print('\033[1m' + '\033[4m' "classification_report Apr_May Trained ,","Tested on Apr_May",'\033[0m'+ '\n','\n',report_def_2m)

test_predict = model_rf.predict(test_june[svar])

predict_def = pd.DataFrame({'target':test_june['target'],
                              'predict':test_predict})


report_def_2m = classification_report(predict_def['target'], predict_def['predict'])


print("Variables used:",'\n',svar,'\n')

print('\n')
print(default_model)
print('\n')
print('\033[1m' + '\033[4m' "classification_report June ,",'\033[0m'+ '\n','\n',report_def_2m)
