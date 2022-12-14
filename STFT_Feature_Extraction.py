#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
# from librosa.feature import mfcc
# from librosa.display import specshow
# from librosa.util import normalize
# from sklearn.preprocessing import StandardScaler
# from librosa.feature import spectral_centroid
# from librosa.feature import spectral_bandwidth
# from sklearn.decomposition import PCA
from scipy.signal import decimate

# stft = pad(np.abs(librosa.stft(y_truncated, num_ffts=225, sample_skips=512)), 128, 1000))
# centroid = librosa.feature.spectral_centroid(y=y_truncated, sr=sr)
# stft_chroma = librosa.feature.chroma_stft(y=y_truncated, sr=sr)
# spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_truncated, sr=sr)


# In[24]:


root = 'C:/Users/Joseph/Downloads/'
X_train = np.load(root+'train_data_matrix_mono_ds=2.npy')
y_train = np.load(root+'train_labels_mono.npy').squeeze()
SAMPLE_RATE = 22050
n, p = X_train.shape


# In[ ]:


# X_train = normalize(X_train, axis=1)
# X_test = normalize(X_test, axis=1)
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# scaler = StandardScaler().fit(X_test)
# X_test = scaler.transform(X_test)


# In[25]:


X_train.shape


# In[27]:


N_FFT = 1024
X_example = np.abs(librosa.stft(X_train[0], n_fft=N_FFT, hop_length=int(N_FFT/2)))
fig, ax = plt.subplots()

img = librosa.display.specshow(librosa.amplitude_to_db(X_example,

                                                       ref=np.max),

                               y_axis='log', x_axis='time', ax=ax)

ax.set_title('Power spectrogram')

fig.colorbar(img, ax=ax, format="%+2.0f dB")


# In[28]:


N_FFT = 1024
X_stft_train = np.abs(librosa.stft(X_train, n_fft=N_FFT, hop_length=int(N_FFT/2)))
print(X_stft_train.shape)


# In[29]:


np.save(root+'X_stft_mono_train_nfft=1024.npy', X_stft_train)

