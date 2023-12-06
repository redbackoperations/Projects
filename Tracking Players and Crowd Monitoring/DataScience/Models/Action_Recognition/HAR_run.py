# from your_sensor_class import SensorHandler  # Assuming the class is in a separate module
import time
import numpy as np
from tensorflow.keras.models import  load_model
from madgwickahrs import MadgwickAHRS
import numpy as np

#Warning: This code is not tested and is only meant to be a guide

class FeatureExtractor:
    def __init__(self):
        self.madgwick_filter = MadgwickAHRS(sampleperiod=1/60)

    def process_batch(self, gyro_readings, accel_readings):
        attitude_roll, attitude_pitch, attitude_yaw = [], [], []
        gravity_x, gravity_y, gravity_z = [], [], []
        rotation_rate_x, rotation_rate_y, rotation_rate_z = [], [], []
        user_acceleration_x, user_acceleration_y, user_acceleration_z = [], [], []

        for gyro, accel in zip(gyro_readings, accel_readings):
            self.madgwick_filter.update_imu(gyro, accel)
            roll, pitch, yaw = self.madgwick_filter.quaternion.to_euler123()
            
            attitude_roll.append(roll)
            attitude_pitch.append(pitch)
            attitude_yaw.append(yaw)
            
            # Apply a low-pass filter to isolate gravity
            gravity = low_pass_filter(accel)
            gravity_x.append(gravity[0])
            gravity_y.append(gravity[1])
            gravity_z.append(gravity[2])

            # Gyro readings are the rotation rates
            rotation_rate_x.append(gyro[0])
            rotation_rate_y.append(gyro[1])
            rotation_rate_z.append(gyro[2])

            # Subtract gravity to get user acceleration
            user_acceleration = accel - gravity
            user_acceleration_x.append(user_acceleration[0])
            user_acceleration_y.append(user_acceleration[1])
            user_acceleration_z.append(user_acceleration[2])

        features = np.column_stack((attitude_roll, attitude_pitch, attitude_yaw,
                                   gravity_x, gravity_y, gravity_z,
                                   rotation_rate_x, rotation_rate_y, rotation_rate_z,
                                   user_acceleration_x, user_acceleration_y, user_acceleration_z))

        return features

# Low-pass filter implementation
def low_pass_filter(signal, alpha=0.8):
    gravity = np.zeros_like(signal)
    gravity = alpha * gravity + (1 - alpha) * signal
    return gravity


class PredictivePipeline:
    def __init__(self, model_path='model.h5', scaler=None):
        self.model = load_model(model_path)
        self.sensor_handler = SensorHandler()
        self.scaler = scaler

    def grab_batch(self, timewindow):
        gyro_readings = []
        accel_readings = []
        start_time = time.time()
        while time.time() - start_time < timewindow:
            gyro, accel = self.sensor_handler.get_readings()
            gyro_readings.append(gyro)  # Assuming gyro is a tuple (x, y, z)
            accel_readings.append(accel) # Assuming accel is a tuple (x, y, z)
        return np.array(gyro_readings), np.array(accel_readings)

    def process_batch(self, batch):
        gyro_readings, accel_readings = batch
        
        # Here you would perform any feature extraction and/or transformation needed to convert the
        # gyroscope and accelerometer readings into the model-usable format.
        # For simplicity, we are just concatenating the two.
        processed_batch = np.concatenate((gyro_readings, accel_readings), axis=1)
        
        # Rescale using the scaler fitted during training
        if self.scaler:
            processed_batch = self.scaler.transform(processed_batch)

        return processed_batch

    def predict_batch(self, processed_batch):
        # Assuming processed_batch is a 2D array with shape (sequence_length, number_of_features)
        processed_batch = np.expand_dims(processed_batch, axis=0)
        prediction = self.model.predict(processed_batch)
        predicted_label = np.argmax(prediction)
        return predicted_label