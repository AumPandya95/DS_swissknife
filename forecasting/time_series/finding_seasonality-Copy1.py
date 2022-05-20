
# coding: utf-8

# In[1]:


import pandas as pd
import timesynth as ts
import matplotlib.pyplot as plt
import scipy.fftpack
import numpy as np


# ### Trying to predict the seasonality of Indian temperature data since 1901.

# In[2]:


temp = pd.read_csv('D:\PGDBA\Assignments_Projects\IMD data\TemperatureMax1.csv')


# In[3]:


temp = temp.set_index('Year')


# In[4]:


temp['Temperature_in_Degrees'].values[:576]


# In[5]:


plt.plot(temp.index[:150], temp['Temperature_in_Degrees'][:150])
plt.show()


# In[6]:


tempy = scipy.fftpack.fft(temp['Temperature_in_Degrees'].values[:576])
freqs = scipy.fftpack.fftfreq(len(tempy),1) # t[1] - t[0])

fig, ax = plt.subplots()

ax.stem(freqs, np.abs(tempy))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-0.3 / 2, 0.3 / 2)
ax.set_ylim(-5, 10000)


# In[7]:


pd.DataFrame(abs(tempy))


# In[8]:


kk = pd.DataFrame(columns = ['y', 'freq'])
kk['y'] = abs(tempy)
kk['freq'] = freqs


# #### As the timeseries has a very small trend, the zeroth value (highest power value) will have a frequency of 0 Hz.
# 
# #### Thus, the seasonality of the Indian temperature cycle captured through this method is 12.

# In[9]:


1/abs(kk.sort_values(by='y', ascending=False).iloc[1].values[1])


# ### Trying to test this method on reproduced timeseries.
# 
# #### Initialising a time series to reproduce a seasonality of 6 months and finding the same through FFT.

# In[10]:


# Initializing TimeSampler
time_sampler = ts.TimeSampler(stop_time=576)
# Sampling irregular time samples
regular_time_samples = time_sampler.sample_regular_time(num_points=576) #,resolution=0.001736) #, keep_percentage=50) , resolution=100, num_points=1000)
# Initializing Sinusoidal signal
sinusoid = ts.signals.Sinusoidal(frequency=0.166)
# Initializing Gaussian noise
white_noise = ts.noise.GaussianNoise(std=0.09)
# Initializing TimeSeries class with the signal and noise objects
timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
# Sampling using the irregular time samples
samples, signals, errors = timeseries.sample(regular_time_samples)


# In[11]:


plt.plot(samples[:18])
plt.show()


# In[12]:


#Y    = np.fft.fft(samples)
y = scipy.fftpack.fft(samples)
freqs = scipy.fftpack.fftfreq(len(y)) #* 576

fig, ax = plt.subplots()

ax.stem(freqs, np.abs(y))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-5, 10)

# plt.figure()
# plt.plot( freq, np.abs(Y) )
# plt.figure()
# plt.plot(freq,  np.angle(Y) )
# plt.show()


# In[13]:


power_freq_dataframe = pd.DataFrame(columns=['power', 'frequency'])


# In[14]:


power_freq_dataframe['power'] = abs(y)
power_freq_dataframe['frequency'] = freqs


# In[15]:


power_freq_dataframe.sort_values(by='power', ascending=False)


# In[16]:


1/abs(power_freq_dataframe.sort_values(by='power', ascending=False).iloc[0].values[1])


# #### Initialising a time series with various standard deviations to reproduce a seasonality of 6 months and finding the same through FFT.

# In[17]:


# Initializing TimeSeries
time_sampler = ts.TimeSampler(stop_time=576)
# Sampling regular time samples
regular_time_samples = time_sampler.sample_regular_time(num_points=1728) #,resolution=0.001736) #, keep_percentage=50) , resolution=100, num_points=1000)
# Initializing Sinusoidal signal
sinusoid = ts.signals.Sinusoidal(frequency=0.166)
# Initializing Gaussian noise. This can be adjusted, however, the results are the same.
white_noise = ts.noise.GaussianNoise(std=0.9)
# Initializing TimeSeries class with the signal and noise objects
timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
# Sampling using the regular time samples
samples1, signals, errors = timeseries.sample(regular_time_samples)


# In[18]:


plt.plot(samples1[:])
plt.show()


# In[19]:


#Y    = np.fft.fft(samples)
y1 = scipy.fftpack.fft(samples1)
freqs1 = scipy.fftpack.fftfreq(len(y1))*3 #* 576

fig, ax = plt.subplots()

ax.stem(freqs1, np.abs(y1))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-5, 150)

# plt.figure()
# plt.plot( freq, np.abs(Y) )
# plt.figure()
# plt.plot(freq,  np.angle(Y) )
# plt.show()


# In[20]:


power_freq_dataframe1 = pd.DataFrame(columns=['power1', 'frequency1'])
power_freq_dataframe1['power1'] = abs(y1)
power_freq_dataframe1['frequency1'] = freqs1


# #### Thus, the seasonality comes out to be 6 through this method.

# In[21]:


1/abs(power_freq_dataframe1.sort_values(by='power1', ascending=False).iloc[0].values[1])

