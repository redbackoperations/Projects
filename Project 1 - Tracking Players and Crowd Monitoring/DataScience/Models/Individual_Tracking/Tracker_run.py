import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from geopy import distance
from geopy.distance import geodesic
from geopy.distance import great_circle
import numpy as np
import glob
import os
from datetime import datetime,timedelta
from tensorflow.keras.models import load_model
from PMT_run import PredictiveTracking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error,mean_absolute_error


import sys

sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')

broker_address="test.mosquitto.org"
topic="Orion_test/Individual Tracking & Monitoring"


topic="Orion_test/UTP"

from Dashboard import Dashboard as DB

from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH 
Handler=MQDH(broker_address, topic)
Reciever=MQDH(broker_address, "Orion_test/UTP")

class RealTimeTracking:
    def __init__(self,user_id):
        """
        Initializes the RealTimeTracking class with user_id.
        :param user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.seq_length = 20
        self.train_date=None
        self.model_path = f"Models/Individual Tracking & Monitoring/IndiMods/model_{user_id}.h5"
        self.predictive_model=None
       

    def get_trajectory(self, gps_data):
        """
        Filters the GPS data to return only the trajectory for the given user.
        :param gps_data: DataFrame containing GPS data
        :return: DataFrame containing the trajectory for the user
        """
        return gps_data[gps_data['user_id'] == self.user_id]

    def get_direction(self, point1, point2):
        """
        Calculates the direction from point1 to point2.
        :param point1: Dictionary containing the 'Latitude' and 'Longitude' of the first point
        :param point2: Dictionary containing the 'Latitude' and 'Longitude' of the second point
        :return: Direction angle in radians
        """

        return np.arctan2(point2['Longitude'] - point1['Longitude'], point2['Latitude'] - point1['Latitude'])

    def get_distance(self, point1, point2):
        """
        Calculates the distance between two geographical points.
        :param point1: Dictionary containing the 'Latitude' and 'Longitude' of the first point
        :param point2: Dictionary containing the 'Latitude' and 'Longitude' of the second point
        :return: Distance in meters
        """
        return distance.distance((point1['Latitude'],point1['Longitude']),(point2['Latitude'],point2['Longitude'])).meters

    def get_speed(self, initialpoint,finalpoint,initialtime,finaltime):
        """
        Calculates the speed between two points given the time taken to travel.
        :param initialpoint: Starting point as a dictionary with 'Latitude' and 'Longitude'
        :param finalpoint: Ending point as a dictionary with 'Latitude' and 'Longitude'
        :param initialtime: Start time as a datetime object
        :param finaltime: End time as a datetime object
        :return: Speed in meters per second
        """
        return self.get_distance(initialpoint,finalpoint) / (finaltime - initialtime).seconds


    def get_acceleration(self, initialspeed,finalspeed,initialtime,finaltime):       
        """
        Calculates the acceleration given the initial and final speeds and the time taken.
        :param initialspeed: Initial speed in meters per second
        :param finalspeed: Final speed in meters per second
        :param initialtime: Start time as a datetime object
        :param finaltime: End time as a datetime object
        :return: Acceleration in meters per second squared
        """ 

        return (finalspeed - initialspeed) / (finaltime - initialtime).seconds
    
    def get_stops(self, trajectory, time_threshold):
        stops = []
        for i in range(1, len(trajectory)):
            if self.get_distance(trajectory.iloc[i-1], trajectory.iloc[i]) == 0 and \
            (trajectory.iloc[i]['Datetime'] - trajectory.iloc[i-1]['Datetime']).seconds >= time_threshold:
                stops.append(trajectory.iloc[i-1])
        return stops

    def get_mode(self, speeds, accelerations):
        avg_speed = np.mean(speeds)
        avg_acceleration = np.mean(accelerations)
        if avg_speed < 2:
            return 'walking'
        elif avg_speed < 20:
            return 'cycling'
        else:
            return 'driving'
        


    def get_frequent_areas(self, trajectory, eps=0.01, min_samples=2):
        coords = [point[['Latitude', 'Longitude']].tolist() for _, point in trajectory.iterrows()]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label != -1:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(trajectory.iloc[i])
        return clusters

    def get_anomalies(self, trajectory):
        coords = [point[['Latitude', 'Longitude']].tolist() for _, point in trajectory.iterrows()]
        model = IsolationForest().fit(coords)
        anomalies = [point for i, point in enumerate(trajectory.iterrows()) if model.predict([coords[i]]) == -1]
        return anomalies

    def get_time_feature(self, cluster):
        return cluster.iloc[0]['Datetime'].hour
    
    def create_prediction_data(self, trajectory, common_areas=None):
        # Process the data similarly to the preprocess_data method
        X, _ = self.preprocess_data(trajectory, common_areas)
        
      

        # Select one sequence
        pred_sequence = X[0]

        # Replace the first four features with placeholder values (e.g., zeros)
        pred_sequence[:, :4] = 0

        # Reshape to match the expected input shape
        pred_sequence = pred_sequence.reshape(1, pred_sequence.shape[0], pred_sequence.shape[1])

        return pred_sequence             
                           
    def get_ground_truth(self, trajectory):
        # Preprocess the data to get the original features, including the coordinates
        _, y = self.preprocess_data(trajectory)

        # Select the corresponding ground truth for the first sequence
        ground_truth = y[0]

        return ground_truth                        
        
    def test_prediction(self, trajectory, common_areas=None):
        # Create test data for the given trajectory
        test_sequence = self.create_prediction_data(trajectory, common_areas)

        # Load the trained model
        try:
            self.model = load_model(self.model_path)
        except:
            print("No model found. Please train the model first.")
            return
        # Make a prediction using the test sequence
        prediction = self.model.predict(test_sequence)

        # Retrieve the ground truth (actual future coordinates) for the test sequence
        # Assuming you have a method to obtain the ground truth
        ground_truth = self.get_ground_truth(trajectory)

        # Compute metrics
        mae = mean_absolute_error(ground_truth, prediction[0])
        mse = mean_squared_error(ground_truth, prediction[0])

        # Print the prediction results
        print("Predicted coordinates:")
        for coords in prediction[0]:
            print(f"Latitude: {coords[0]}, Longitude: {coords[1]}")

        # Print the metrics
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        
    def preprocess_data(self, trajectory, common_areas=None):
        frequent_areas = list(self.get_frequent_areas(trajectory, 10, 5).values())
        combined_areas = [area[0] for area in frequent_areas[:10]]  # Take the first 10 frequent areas

        # If there are fewer than 10 frequent areas, add common areas
        while len(combined_areas) < 10:
            if common_areas is not None and len(common_areas) > 0:
                combined_areas.append({'Latitude': common_areas[0][0], 'Longitude': common_areas[0][1]})
                common_areas.pop(0)
            else:
                combined_areas.append({'Latitude': 0, 'Longitude': 0})  # Fill with zeros if not enough areas

        features = []
        for i, point in enumerate(trajectory.iloc[:-1].to_dict('records')):
            # Add the coordinates to the feature list along with other features
            feature_vector = [
                point['Latitude'],
                point['Longitude'],
                self.get_speed(point, trajectory.iloc[i+1], point['Datetime'], trajectory.iloc[i+1]['Datetime']),
                self.get_direction(point, trajectory.iloc[i+1]), # Direction
                point['Datetime'].weekday(),
                point['Datetime'].hour,
                point['Datetime'].minute
            ] + [area['Latitude'] for area in combined_areas] + [area['Longitude'] for area in combined_areas]

            features.append(feature_vector)

        # Create sequences and future coordinates
        sequences = []
        future_coordinates = []
        for i in range(len(features) - self.seq_length - 9):  # 9 is to accommodate 10 future coordinates
            sequence = features[i:i+self.seq_length]
            future_coord = [features[j][:2] for j in range(i+self.seq_length, i+self.seq_length+10)]  # Take only Latitude and Longitude
            sequences.append(sequence)
            future_coordinates.append(future_coord)

        # Convert to NumPy arrays
        X = np.array(sequences)
        y = np.array(future_coordinates)

        return X, y


    def generate_future_coordinates(self, trajectory, common_areas=None, number_of_future_points=100, time_interval_minutes=5):
        print(trajectory.head(-5))
        # Get the latest data point from the trajectory
        latest_data_point = trajectory.iloc[-1]

        # Create a list of future datetimes
        future_datetimes = [latest_data_point['Datetime'] + timedelta(minutes=i * time_interval_minutes) for i in range(1, number_of_future_points + 1)]

        # Create a DataFrame to hold the future trajectory
        future_trajectory = pd.DataFrame(columns=trajectory.columns)

        # Populate the future trajectory DataFrame with the latest data point and future datetimes
        for future_datetime in future_datetimes:
            future_data_point = latest_data_point.copy()
            future_data_point['Datetime'] = future_datetime
            future_trajectory = future_trajectory.append(future_data_point, ignore_index=True)
        print(future_trajectory.head(5))
        # Create test data for the given trajectory and future datetimes
        pred_sequence = self.create_prediction_data(future_trajectory, common_areas)

        # Load the trained model
        try:
            self.model = load_model(self.model_path)
        except:
            print("No model found. Please train the model first.")
            return

        # Make a prediction using the test sequence
        prediction = self.model.predict(pred_sequence)

        # Convert the prediction to a list of coordinates (latitude, longitude)
        predicted_coordinates = [(coords[0], coords[1]) for coords in prediction[0]]

        return predicted_coordinates


    def train_personalised_model(self, trajectory_data, retrain=False):        
        preprocessed_data = self.preprocess_data(trajectory_data)
        
        try:       
            model=load_model(self.model_path)
            
        except:
            model=None
        if retrain and model is not None:
            print("Retraining model......")
            self.predictive_model = PredictiveTracking(self.user_id, preprocessed_data,'train')
            self.predictive_model.train_model()             
            self.predictive_model.save_model()
        
            return
        elif not retrain and model is not None:
            print("Model already trained. Use retrain=True to retrain the model.")
            return
        
        elif model is None:
            print("No model found. Training new model...")
            self.predictive_model = PredictiveTracking(self.user_id, preprocessed_data,'train')
            self.predictive_model.train_model()
            self.predictive_model.save_model()
            return











# this should only be used during testing --some modifications needed to use it in production
def read_plt(file_path, user_id):
    """
    Reads a plt file and returns it as a DataFrame, adding a user_id column.
    :param file_path: Path to the plt file
    :param user_id: Unique identifier for the user
    :return: DataFrame containing the plt file data with added user_id
    """
    columns = ['Latitude', 'Longitude', 'Reserved', 'Altitude', 'NumDays', 'Date', 'Time']
    data = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
    data['Altitude'] = data['Altitude'] * 0.3048
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.drop(columns=['Date', 'Time'], inplace=True)
    data['user_id'] = user_id # Add user_id to the DataFrame

    return data







# common_areas = [(lat1, lon1), (lat2, lon2), ...] # Example common areas i.e. landmarks,coffee shops








'''-----------------------------------------------------------------------------------------------------------------------------------------------------------the number 00x is the user id, it should be changed to the user id of the user whose data is being processed'''
directory_path = r"E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Clean Datasets\Geolife Trajectories 1.3\Data\021\Trajectory\*.plt"

# Extract the user ID from the directory path (assuming it's the parent folder of "Trajectory")
user_id = os.path.basename(os.path.dirname(os.path.dirname(directory_path)))

# Initialize the RealTimeTracking object with the extracted user ID
real_time_tracking = RealTimeTracking(user_id)

# Loop through the .plt files and process the trajectory data
for file_path in glob.glob(directory_path):
    trajectory_data = read_plt(file_path, user_id)
    user_trajectory = real_time_tracking.get_trajectory(trajectory_data)
    
# Train the model
real_time_tracking.train_personalised_model(user_trajectory,False)
print(user_id)

# Predict the future coordinates
predicted_coordinates = real_time_tracking.generate_future_coordinates(user_trajectory)
print(predicted_coordinates)

#data=Handler.receive_data()
#user_id=dReciever.receive_data()


Handler.send_data(predicted_coordinates, user_id)