import pandas as pd
import os
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split. LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn import metrics

#create columns for the dataframe
columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv("WISDM_ar_v1.1_raw.txt", header = None, names = columns, on_bad_lines = "skip")


#Make sure all the values are the same time and extra variables are not added into the data - remove ";"
df['z-axis'].replace(';', '', inplace = True, regex = True)
#drop any NaN
df.dropna(inplace=True)

#Make all axis same data type
df['x-axis'] = df['x-axis'].astype('float64')
df['y-axis'] = df['y-axis'].astype('float64')
df['z-axis'] = df['z-axis'].astype('float64')


# In[29]:


##Extract Features
warnings.filterwarnings('ignore')

def get_time_series_features(signal):
    window_size = len(signal)
    # mean
    sig_mean = np.mean(signal)
    # standard deviation
    sig_std = np.std(signal)
    # avg absolute difference
    sig_aad = np.mean(np.absolute(signal - np.mean(signal)))
    # min
    sig_min = np.min(signal)
    # max
    sig_max = np.max(signal)
    # max-min difference
    sig_maxmin_diff = sig_max - sig_min
    # median
    sig_median = np.median(signal)
    # median absolute deviation
    sig_mad = np.median(np.absolute(signal - np.median(signal)))
    # Inter-quartile range
    sig_IQR = np.percentile(signal, 75) - np.percentile(signal, 25)
    # negative count
    sig_neg_count = np.sum(s < 0 for s in signal)
    # positive count
    sig_pos_count = np.sum(s > 0 for s in signal)
    # values above mean
    sig_above_mean = np.sum(s > sig_mean for s in signal)
    # number of peaks
    sig_num_peaks = len(find_peaks(signal)[0])
    # skewness
    sig_skew = stats.skew(signal)
    # kurtosis
    sig_kurtosis = stats.kurtosis(signal)
    # energy
    sig_energy = np.sum(s ** 2 for s in signal) / window_size
    # signal area
    sig_sma = np.sum(signal) / window_size

    return [sig_mean, sig_std, sig_aad, sig_min, sig_max, sig_maxmin_diff, sig_median, sig_mad, sig_IQR, sig_neg_count, sig_pos_count, sig_above_mean, sig_num_peaks, sig_skew, sig_kurtosis, sig_energy, sig_sma]


def get_freq_domain_features(signal):
    all_fft_features = []
    window_size = len(signal)
    signal_fft = np.abs(np.fft.fft(signal))
    # Signal DC component
    sig_fft_dc = signal_fft[0]
    # aggregations over the fft signal
    fft_feats = get_time_series_features(signal_fft[1:int(window_size / 2) + 1])

    all_fft_features.append(sig_fft_dc)
    all_fft_features.extend(fft_feats)
    return all_fft_features


# In[30]:


#Go through timestamps, anything greater than zero stays in the data
data = df[df['timestamp'] > 0]


#put time stamps in ascending order
data = data.sort_values(by='timestamp', ascending=True)

#Change from nanoseconds to seconds
user_groups = data.groupby(['user'])

#normalize timestamp to start from 0 - subtract the lowest timestamp from the rest of the times for each user
#build new column of normalized timestamps
data['time'] = user_groups['timestamp'].apply(lambda x: (x - x.min()) / 1000000000) 


# In[32]:


#Convert signals to magnitude to remove orientation
#Take each value from x-array, y-array, and z-array, add them up to make one data point for magntiude
data['x-axis'] = df['x-axis']**(2)
data['y-axis'] = df['y-axis']**(2)
data['z-axis'] = df['z-axis']**(2)

#Take the square root and add a new column called "magnitude" to the dataframe 
data["magnitude"] = data["x-axis"]+data["y-axis"]+data["z-axis"]
data["magnitude"] = np.sqrt(np.abs(data["magnitude"]))


# In[172]:


#Segmenting the data

#iterate into the data, for every 200 readings, take the magnitude readings from those time and see the most common frequency. 
#Add that into an array of what feature to the frequency found 
 
x=0
seg = pd.DataFrame(columns = ['segments', 'acts', 'user', 'features'])
for user in range(data['user'].min(), data['user'].max()):
    user_df = data[data['user'] == user]   
    
    #step size is 100 so we can create overlap within the data points and cover transitional states
    for i in range(0, len(user_df), 100):
        mag = user_df['magnitude'].values[i: i+ 200]
        
        #even consider if the magnitude length is shorter than 200
        if len(mag) < 200:
            continue
        
        #get the label for the activity for every 200 data points - find the mode 
        labels = user_df['activity'][i: i+ 200]
        unique, counts = np.unique(labels, return_counts=True)
        label = unique[np.argmax(counts)]
        features = get_time_series_features(mag)
        frequency = get_freq_domain_features(mag)
        features.extend(frequency)
        seg.loc[x] = (mag, label, user, features)
        x=x+1 


# In[173]:


# Convert dataset to format that can be turned into a matrix
#new_dataset = pd.DataFrame(columns=range(37))

# user + activity + number of features = the number of columns
column_len = 2 +len(seg.loc[0].at['features'])

new_dataset = pd.DataFrame(columns = range(column_len))

for i in range(len(seg)):
    matrixList = [seg.loc[i].at['user'], seg.loc[i].at['acts']]
    matrixList.extend(seg.loc[i].at['features'])
    new_dataset.loc[i] = matrixList


# In[174]:


####Train Classifiers - Train and Test 

#a train/test split with the first 29 users are training and last 7 users are testing
train_dataset = new_dataset[new_dataset[0] <= 29]
test_dataset = new_dataset[new_dataset[0] > 29]

X_train = train_dataset.drop(columns = [0, 1])
X_test = test_dataset.drop(columns = [0, 1])

y_train = train_dataset[1]
y_test = test_dataset[1]


# In[175]:


#Build model and predictions based off the training and test sets 
model = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)
preds = model.predict(X_test)
model_proba = model.predict_proba(X_test)


# In[176]:


##Results for Train/Test Split
class_report_testtrain =classification_report(y_test, preds)
print('***** Results for Train/Test Split *****')
print('Accuracy Score:', metrics.accuracy_score(y_test, preds))

print('AUROC Score', roc_auc_score(y_test, model_proba, multi_class='ovr'))
print('          ')
print('Classification Report:')
print(class_report_testtrain)

#confusion matrix for test/training 
confusion_matrix_testtrain = metrics.confusion_matrix(y_test, preds)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_testtrain, display_labels = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs","Walking"])
cm_display.plot()
plt.show()


# In[177]:


##Training classifiers -- 10 fold cross validation 

X = new_dataset.drop(columns = [0,1])
y = new_dataset[1]

#Using linear repression model, test for cv = 10
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

#get predictions and probabilities 
CV10_preds = cross_val_predict(model, X, y, cv=10) #returns label predictions
CV10_proba = cross_val_predict(model, X, y, cv=10, method='predict_proba')

CV10_score = cross_val_score(model, X, y, cv = 10)


# In[178]:


##Results for 10-fold CV
class_report_CV10 = classification_report(y, CV10_preds)
print('***** Results for 10-fold CV *****')
print('Accuracy Score:', metrics.accuracy_score(y, CV10_preds))  

print('AUROC Score', roc_auc_score(y, CV10_proba, multi_class='ovr'))
print('          ')
print('Classification Report:')
print(class_report_CV10)

confusion_matrix_CV10 = metrics.confusion_matrix(y, CV10_preds)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_CV10, display_labels = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs","Walking"])
cm_display.plot()
plt.show()


# In[179]:


#Training classifiers - Leave one group out

#get just the groups 
users = new_dataset[0]

lo = LeaveOneGroupOut
lo_splits = lo.get_n_splits(X, y, groups = users)

logo_preds = cross_val_predict(model, X, y, cv=lo_splits)
logo_proba = cross_val_predict(model, X, y, cv = lo_splits, method = "predict_proba")


# In[180]:


##Results for Leave One Group Out CV
class_report_logo =classification_report(y, logo_preds)

print('***** Results for Leave-One-Group-Out CV *****')
print('Accuracy Score:', metrics.accuracy_score(y, logo_preds))  

print('AUROC Score', roc_auc_score(y, logo_proba, multi_class='ovr'))
print('          ')
print('Classification Report:')
print(class_report_logo)

confusion_matrix_logo = metrics.confusion_matrix(y, logo_preds)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_logo, display_labels = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs","Walking"])
cm_display.plot()
plt.show()

