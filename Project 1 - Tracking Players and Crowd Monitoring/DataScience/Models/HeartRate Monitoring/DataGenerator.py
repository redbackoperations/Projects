from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import random

def generate_ppg_data(number_of_individuals, sample_frequency, duration):
    """
    Generate synthetic PPG data for simulating both healthy and unhealthy individuals.

    :param number_of_individuals: int, the number of individual PPG data sets to create
    :param sample_frequency: float, the sampling frequency in Hz
    :param duration: float, the total duration in seconds for each individual's PPG data
    """
    
    # Constants for mean heart rates of healthy and unhealthy conditions
    healthy_mean_hr = 75
    unhealthy_mean_hr = 90
    noise_level = 0.05  # Noise level for the generated signal

    for individual in range(number_of_individuals):
        # Randomly determine the start date and time for the simulation
        start_datetime = datetime(random.randint(2019,2023), random.randint(1,12), random.randint(1,28), random.randint(0,23), random.randint(0,59), random.randint(0,59))
        
        # Prepare the CSV data
        csv_data = []
        columns = ['DayOfWeek', 'Hour', 'Minute', 'Second', 'Condition', 'PPG_Signal']
        condition = 'Healthy' if np.random.rand() > 0.2 else 'Unhealthy'  # Assign condition with 20% probability of being unhealthy
        mean_heart_rate = healthy_mean_hr if condition == 'Healthy' else unhealthy_mean_hr
        
        # Generate time intervals
        time_intervals = [start_datetime + timedelta(seconds=t) for t in np.arange(0, duration, 1 / sample_frequency)]

        # Generate baseline and heartbeat signals
        baseline_signal = np.sin(2 * np.pi * mean_heart_rate / 60 * np.arange(0, duration, 1 / sample_frequency))
        heartbeat_signal = np.sin(2 * np.pi * mean_heart_rate / 60 * 10 * np.arange(0, duration, 1 / sample_frequency))
        peaks, _ = find_peaks(heartbeat_signal, height=0.5)
        heartbeat_signal *= 0
        heartbeat_signal[peaks] = 1

        # Combine signals and add noise
        ppg_signal = baseline_signal + heartbeat_signal
        ppg_signal += np.random.normal(0, noise_level, len(ppg_signal))

        # Append to CSV data with DateTime format
        for datetime_interval, ppg in zip(time_intervals, ppg_signal):
            day_of_week = datetime_interval.isoweekday()  # Monday is 1, Sunday is 7
            hour = datetime_interval.hour
            minute = datetime_interval.minute
            second = datetime_interval.second
            csv_data.append([day_of_week, hour, minute, second, condition, ppg])

        # Create a DataFrame
        df = pd.DataFrame(csv_data, columns=columns)

        # Save to CSV file
        file_name = f"E:\\Dev\\Deakin\\Project_Orion\\DataScience\\Models\\HeartRate Monitoring\\HRD\\individual_{individual}.csv"
        df.to_csv(file_name, index=False)

# Example usage to generate individual CSV files
generate_ppg_data(number_of_individuals=100, sample_frequency=100, duration=600)
