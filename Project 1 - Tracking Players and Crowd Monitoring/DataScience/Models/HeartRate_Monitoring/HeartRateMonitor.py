import math
import numpy as np
import datetime
import pandas as pd
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis, zscore
from sklearn.preprocessing import StandardScaler
'''
The HRMonitoring class is meant to manage the data input and output of the system. It is the main class that calls the other classes and functions.

'''
class HRmonitoring():
    def __init__(self,userID):
        self.userID = userID
        self.data = None
        self.features = None
        self.model = None
        self.prediction = None
        self.analysis = None
        self.mode = None
    def get_data(self, source, mode='offline'):
        # Fetch data from the source
        # For demonstration purposes, assuming data is fetched into a DataFrame
        self.data = pd.read_csv(source)
        self.mode = mode
        
    def create_sequences(data, window_size):
        sequences = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i+window_size])
        return np.array(sequences)
    
    
    def extract_features(self):
        # Ensure data is present
        if self.data is None:
            print("Data not loaded.")
            return

        # Baseline Heart Rate
        baseline_hr = np.mean(self.data['heart_rate'])

        # Heart Rate Variability (HRV)
        rr_intervals = np.diff(self.data['heart_rate'].values)  # Difference between successive heart rates
        hrv = np.std(rr_intervals)

        # Outliers using IQR
        Q1 = np.percentile(self.data['heart_rate'], 25)
        Q3 = np.percentile(self.data['heart_rate'], 75)
        IQR = Q3 - Q1
        outlier_count = np.sum((self.data['heart_rate'] < (Q1 - 1.5 * IQR)) | (self.data['heart_rate'] > (Q3 + 1.5 * IQR)))

        # Frequency Domain Features using Fourier Transform
        yf = fft(self.data['heart_rate'])
        power_spectrum = np.abs(yf)
        dominant_frequency = np.argmax(power_spectrum)

        # Z-score compared to all users (assuming self.data contains data from all users)
        user_mean = np.mean(self.data[self.data['userID'] == self.userID]['heart_rate'])
        overall_mean = np.mean(self.data['heart_rate'])
        overall_std = np.std(self.data['heart_rate'])
        z = (user_mean - overall_mean) / overall_std

        # Moving Average (7-day window as an example)
        moving_avg = self.data['heart_rate'].rolling(window=7).mean()

        # Skewness and Kurtosis
        skewness = skew(self.data['heart_rate'])
        kurt = kurtosis(self.data['heart_rate'])

        # Store features
        self.features = {
            'baseline_hr': baseline_hr,
            'hrv': hrv,
            'outlier_count': outlier_count,
            'dominant_frequency': dominant_frequency,
            'z_score': z,
            'moving_avg': moving_avg,
            'skewness': skewness,
            'kurtosis': kurt
        }

    
    def process(mode='predict'):
        pass
    def predict():
        pass
    def analyse():
        pass