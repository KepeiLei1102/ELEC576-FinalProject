#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from scipy import fftpack
from scipy.signal import decimate

root = 'C:/Users/Joseph/Downloads/'
trainpath = root + 'TrainRaw/'
# testpath = root + 'TestRaw/'


# In[37]:


# def create_matrix(path, train = True):
filenames = sorted([f for f in os.listdir(trainpath) if f[-3:]=='wav' and f[-5]=='b'])
print(len(filenames))
dec_factor = 2
X = np.zeros((len(filenames),int(441000/dec_factor)))
y = np.zeros((len(filenames),1)) - 1
label_dict = {'airport': 0, 'bus': 1, 'metro': 2}
lengths = []

for i, f in enumerate(filenames):
    samplerate, data = wavfile.read(trainpath+f)
#     print(data.shape)
    data = decimate(data, dec_factor)
    lengths.append(len(data))

    X[i,:len(data)] = data
    label = f.split('-')[0]
    y[i] = label_dict[label]
#     if i%20 == 0:
    print(f)

# np.save(root+'train_data_matrix_ds=2', X)
# np.save(root+'train_labels', y)

# create_matrix(trainpath, True)
# create_matrix(testpath, False)


# In[38]:


np.save(root+'train_data_matrix_mono_ds=2', X)
np.save(root+'train_labels_mono', y)


# In[ ]:


# X_train_ds2 = np.load(root+'train_data_matrix_ds=2.npy')
# np.save(root+'train_data_matrix_2e5samples', X_train[:, :200000])
# X_train_ds2_fragmented = np.copy(X_train_ds2[:,:300000].reshape((2100,100000)))
# np.save(root+'train_data_matrix_2e5samples_3obsPerWav_ds=2', X_train_ds2_fragmented)
# %reset_selective X_train

# X_test_ds2 = np.load(root+'test_data_matrix_ds=2.npy')
# np.save(root+'test_data_matrix_2e5samples', X_test[:, :200000])
# X_test_ds2_fragmented = X_test_ds2[:,:300000].reshape((900,100000))
# np.save(root+'test_data_matrix_2e5samples_3obsPerWav_ds=2', X_test_ds2_fragmented)

# y_train = np.load(root+'train_labels.npy')
# np.save(root+'train_labels_3obsPerWav', np.repeat(y_train, 3))

