import math
import numpy as np
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import panel as pn
from scipy.fft import fft
from scipy.stats import skew, kurtosis, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import OneClassSVM
'''
The HRMonitoring class is meant to manage the data input and output of the system. It is the main class that calls the other classes and functions.

'''
class HRmonitoring():
    def __init__(self,userID):
        self.userID = userID
        self.data = None
        self.features = None

    def get_data(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data['userID'] = self.userID


        
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
        baseline_hr = np.mean(self.data['PPG_Signal'])

        # Heart Rate Variability (HRV)
        rr_intervals = np.diff(self.data['PPG_Signal'].values)  # Difference between successive heart rates
        hrv = np.std(rr_intervals)

        # Outliers using IQR
        Q1 = np.percentile(self.data['PPG_Signal'], 25)
        Q3 = np.percentile(self.data['PPG_Signal'], 75)
        IQR = Q3 - Q1
        outlier_count = np.sum((self.data['PPG_Signal'] < (Q1 - 1.5 * IQR)) | (self.data['PPG_Signal'] > (Q3 + 1.5 * IQR)))

        # Frequency Domain Features using Fourier Transform
        yf = fft(self.data['PPG_Signal'].to_numpy())
        power_spectrum = np.abs(yf)
        dominant_frequency = np.argmax(power_spectrum)


     

        # Moving Average (7-day window as an example)
        moving_avg = self.data['PPG_Signal'].rolling(window=7).mean()

        # Skewness and Kurtosis
        skewness = skew(self.data['PPG_Signal'])
        kurt = kurtosis(self.data['PPG_Signal'])

        # Store features
        self.features = {
            'baseline_hr': baseline_hr,
            'hrv': hrv,
            'outlier_count': outlier_count,
            'dominant_frequency': dominant_frequency,
            'moving_avg': moving_avg,
            'skewness': skewness,
            'kurtosis': kurt
        }
        
        


    def detect_spikes(self, window_size=5, threshold=2.0):
        if self.data is None:
            print("Data not loaded.")
            return

        # Calculate rolling mean and standard deviation
        rolling_mean = self.data['PPG_Signal'].rolling(window=window_size).mean()
        rolling_std = self.data['PPG_Signal'].rolling(window=window_size).std()

        # Detect spikes where the signal deviates from the rolling mean by more than the threshold times the rolling standard deviation
        spikes = np.where(np.abs(self.data['PPG_Signal'] - rolling_mean) > threshold * rolling_std)[0]
        
        return spikes

    def detect_shifts(self, window_size=50, threshold=2.0):
        if self.data is None:
            print("Data not loaded.")
            return

        # Calculate rolling mean for two consecutive windows
        rolling_mean_1 = self.data['PPG_Signal'].rolling(window=window_size).mean().shift(-window_size)
        rolling_mean_2 = self.data['PPG_Signal'].rolling(window=window_size).mean()

        # Detect shifts where the difference between the means of two consecutive windows is greater than the threshold
        shifts = np.where(np.abs(rolling_mean_1 - rolling_mean_2) > threshold)[0]
        
        return shifts
    
    def process():


        # Encoding the 'Condition' column
        label_encoder = LabelEncoder()
        df['Condition_encoded'] = label_encoder.fit_transform(df['Condition'])

        # Drop the original 'Condition' column and use the encoded one
        df = df.drop(columns=['Condition'])

        # Splitting the data into training and testing sets
        X = df.drop(columns=['PPG_Signal'])
        y = df['PPG_Signal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train.head()
        
        return X_train, X_test, y_train, y_test

     
    def train(X_train, X_test, y_train, y_test):
                # Training Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predicting on the test set
        rf_predictions = rf_model.predict(X_test)
        # Evaluate the model using Mean Squared Error (MSE)
        rf_mse = mean_squared_error(y_test, rf_predictions)
        
        return rf_model, rf_mse

      
    
    def predict(model, X_test):
        # Predicting on the test set
        predictions = model.predict(X_test)
        return predictions
        
    def train_anomaly_detector(self, sample_size=10000):
        # Sample a subset of the data for training the One-Class SVM
        subset_data = self.data.sample(sample_size, random_state=42)
        
        # Prepare features for the model
        features = subset_data[['PPG_Signal']]
        
        # Train the One-Class SVM
        self.anomaly_model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
        self.anomaly_model.fit(features)

    def detect_anomalies(self):
        if self.anomaly_model is None:
            print("Model not trained. Use the train_anomaly_detector method first.")
            return

        # Predict anomalies on the data
        anomalies = self.anomaly_model.predict(self.data[['PPG_Signal']])
        
        # Filter out the anomaly rows
        anomaly_data = self.data[anomalies == -1]
        
        return anomaly_data
        
        
path=r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\HeartRate_Monitoring\HRD'
monitor = HRmonitoring(userID="individual_0")
monitor.get_data(os.path.join(path, "individual_0.csv"))

# Extract features using the class methods
monitor.extract_features()

# Detect spikes and shifts
spikes = monitor.detect_spikes()
shifts = monitor.detect_shifts()

# Train the anomaly detector and detect anomalies
monitor.train_anomaly_detector()
anomalies = monitor.detect_anomalies()

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
plt.plot(monitor.data['PPG_Signal'], label="PPG Signal", alpha=0.7)
plt.scatter(spikes, monitor.data['PPG_Signal'].iloc[spikes], color='red', label="Detected Spikes", s=50, marker="x")
plt.scatter(shifts, monitor.data['PPG_Signal'].iloc[shifts], color='blue', label="Detected Shifts", s=50, marker="o")
plt.scatter(anomalies.index, anomalies['PPG_Signal'], color='green', label="Detected Anomalies", s=50, marker="s")
plt.legend()
plt.title("PPG Signal with Detected Spikes, Shifts, and Anomalies")
plt.xlabel("Time (Samples)")
plt.ylabel("PPG Signal Value")
plt.tight_layout()
plt.show()

monitor.extract_features()

# Plotting the extracted features
fig, axs = plt.subplots(4, 1, figsize=(15, 15))

# PPG Signal with Moving Average and Outliers
axs[0].plot(monitor.data['PPG_Signal'], label="PPG Signal", alpha=0.7)
axs[0].plot(monitor.features['moving_avg'], label="Moving Average (7-sample window)", color="purple")
Q1 = np.percentile(monitor.data['PPG_Signal'], 25)
Q3 = np.percentile(monitor.data['PPG_Signal'], 75)
IQR = Q3 - Q1
outliers = monitor.data[(monitor.data['PPG_Signal'] < (Q1 - 1.5 * IQR)) | (monitor.data['PPG_Signal'] > (Q3 + 1.5 * IQR))]
axs[0].scatter(outliers.index, outliers['PPG_Signal'], color='red', label="IQR Outliers", s=50, marker="x")
axs[0].set_title("PPG Signal with Moving Average and IQR Outliers")
axs[0].set_xlabel("Time (Samples)")
axs[0].set_ylabel("PPG Signal Value")
axs[0].legend()

# Frequency Domain Representation of PPG Signal
yf = fft(monitor.data['PPG_Signal'].to_numpy())
power_spectrum = np.abs(yf[:len(yf)//2])  # Taking half due to symmetry
frequencies = np.linspace(0, 0.5, len(power_spectrum))  # Frequency values from 0 to Nyquist (0.5 for normalized freq.)
axs[1].plot(frequencies, power_spectrum, label="Frequency Spectrum", alpha=0.7)
axs[1].axvline(frequencies[monitor.features['dominant_frequency']], color='red', linestyle='--', label="Dominant Frequency")
axs[1].set_title("Frequency Domain Representation of PPG Signal")
axs[1].set_xlabel("Frequency (Normalized)")
axs[1].set_ylabel("Magnitude")
axs[1].legend()

# Displaying HRV
axs[2].text(0.5, 0.6, f"Heart Rate Variability (HRV): {monitor.features['hrv']:.2f}", 
           horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes, fontsize=14)
axs[2].axis('off')

# Displaying Skewness and Kurtosis
axs[3].text(0.5, 0.7, f"Skewness: {monitor.features['skewness']:.2f}", 
           horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes, fontsize=14)
axs[3].text(0.5, 0.3, f"Kurtosis: {monitor.features['kurtosis']:.2f}", 
           horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes, fontsize=14)
axs[3].axis('off')

plt.tight_layout()

plt.close(fig)
plot_panel = pn.pane.Matplotlib(fig, tight=True)