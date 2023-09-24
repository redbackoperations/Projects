# Models

# Action_Recognition

*from your_sensor_class import SensorHandler  # Assuming the class is in a separate module*

```python
import time
```

```python
import numpy as np
```

```python
from tensorflow.keras.models import  load_model
```

```python
from madgwickahrs import MadgwickAHRS
```

```python
import numpy as np
```

*Warning: This code is not tested and is only meant to be a guide*

```python
class FeatureExtractor:
```

```python
def __init__(self):
```

```python
self.madgwick_filter = MadgwickAHRS(sampleperiod=1/60)
```

```python
def process_batch(self, gyro_readings, accel_readings):
```

```python
attitude_roll, attitude_pitch, attitude_yaw = [], [], []
```

```python
gravity_x, gravity_y, gravity_z = [], [], []
```

```python
rotation_rate_x, rotation_rate_y, rotation_rate_z = [], [], []
```

```python
user_acceleration_x, user_acceleration_y, user_acceleration_z = [], [], []
```

```python
for gyro, accel in zip(gyro_readings, accel_readings):
```

```python
self.madgwick_filter.update_imu(gyro, accel)
```

```python
roll, pitch, yaw = self.madgwick_filter.quaternion.to_euler123()
```

```python
attitude_roll.append(roll)
```

```python
attitude_pitch.append(pitch)
```

```python
attitude_yaw.append(yaw)
```

*Apply a low-pass filter to isolate gravity*

```python
gravity = low_pass_filter(accel)
```

```python
gravity_x.append(gravity[0])
```

```python
gravity_y.append(gravity[1])
```

```python
gravity_z.append(gravity[2])
```

*Gyro readings are the rotation rates*

```python
rotation_rate_x.append(gyro[0])
```

```python
rotation_rate_y.append(gyro[1])
```

```python
rotation_rate_z.append(gyro[2])
```

*Subtract gravity to get user acceleration*

```python
user_acceleration = accel - gravity
```

```python
user_acceleration_x.append(user_acceleration[0])
```

```python
user_acceleration_y.append(user_acceleration[1])
```

```python
user_acceleration_z.append(user_acceleration[2])
```

```python
features = np.column_stack((attitude_roll, attitude_pitch, attitude_yaw,
```

```python
gravity_x, gravity_y, gravity_z,
```

```python
rotation_rate_x, rotation_rate_y, rotation_rate_z,
```

```python
user_acceleration_x, user_acceleration_y, user_acceleration_z))
```

```python
return features
```

*Low-pass filter implementation*

```python
def low_pass_filter(signal, alpha=0.8):
```

```python
gravity = np.zeros_like(signal)
```

```python
gravity = alpha * gravity + (1 - alpha) * signal
```

```python
return gravity
```

```python
class PredictivePipeline:
```

```python
def __init__(self, model_path='model.h5', scaler=None):
```

```python
self.model = load_model(model_path)
```

```python
self.sensor_handler = SensorHandler()
```

```python
self.scaler = scaler
```

```python
def grab_batch(self, timewindow):
```

```python
gyro_readings = []
```

```python
accel_readings = []
```

```python
start_time = time.time()
```

```python
while time.time() - start_time < timewindow:
```

```python
gyro, accel = self.sensor_handler.get_readings()
```

```python
gyro_readings.append(gyro)
```

*Assuming gyro is a tuple (x, y, z)*

```python
accel_readings.append(accel)
```

*Assuming accel is a tuple (x, y, z)*

```python
return np.array(gyro_readings), np.array(accel_readings)
```

```python
def process_batch(self, batch):
```

```python
gyro_readings, accel_readings = batch
```

*Here you would perform any feature extraction and/or transformation needed to convert the*

*gyroscope and accelerometer readings into the model-usable format.*

*For simplicity, we are just concatenating the two.*

```python
processed_batch = np.concatenate((gyro_readings, accel_readings), axis=1)
```

*Rescale using the scaler fitted during training*

```python
if self.scaler:
```

```python
processed_batch = self.scaler.transform(processed_batch)
```

```python
return processed_batch
```

```python
def predict_batch(self, processed_batch):
```

*Assuming processed_batch is a 2D array with shape (sequence_length, number_of_features)*

```python
processed_batch = np.expand_dims(processed_batch, axis=0)
```

```python
prediction = self.model.predict(processed_batch)
```

```python
predicted_label = np.argmax(prediction)
```

```python
return predicted_label
```

```python
import os
```

```python
import glob
```

```python
from sklearn.preprocessing import MinMaxScaler
```

```python
import wandb
```

```python
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
```

```python
from tensorflow.keras.layers import LSTM,Bidirectional
```

```python
import pandas as pd
```

```python
import numpy as np
```

```python
import tensorflow as tf
```

```python
import sklearn
```

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

```python
from sklearn.model_selection import train_test_split
```

```python
from sklearn.utils import class_weight
```

```python
from tensorflow.keras.models import Sequential
```

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

```python
from tensorflow.keras.layers import Masking,Dense,Dropout
```

```python
from tensorflow.keras.optimizers import Adam
```

```python
import keras
```

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

```python
print("Keras version:", keras.__version__)
```

```python
print("Tensorflow version:", tf.__version__)
```

```python
print("numpy version:", np.__version__)
```

```python
print("pandas version:", pd.__version__)
```

```python
print("scikit-learn version:", sklearn.__version__)
```

The training below is tracked using wandb.ai, the hyperparameters and the model are tracked and can be viewed on the wandb dashboard you can choose not to use wandb by commenting out the wandb.init() and the WandbMetricsLogger and WandbModelCheckpoint callbacks

```python
wandb.init(
```

*set the wandb project where this run will be logged*

```python
project="--",
```

*track hyperparameters and run metadata with wandb.config*

```python
config={
```

```python
"layer_1": 24,
```

```python
"activation_1": "tanh",
```

```python
"layer_2": 24,
```

```python
"activation_2": "tanh",
```

```python
"layer_3": 12,
```

```python
"activation_3": "tanh",
```

```python
"layer_3": 12,
```

```python
"activation_3": "tanh",
```

```python
"optimizer": "adam",
```

```python
"layer_4": 6,
```

```python
"activation_4": "softmax",
```

```python
"loss": "sparse_categorical_crossentropy",
```

```python
"metric": "accuracy",
```

```python
"epoch": 30,
```

```python
"batch_size": 16
```

```python
}
```

```python
)
```

```python
class TrainingClass:
```

```python
def __init__(self, data_path=r'E:\Test\A_DeviceMotion_data'):
```

```python
self.data_path = data_path
```

```python
self.model = None
```

```python
self.scaler = None
```

```python
self.X_train = None
```

```python
self.X_test = None
```

```python
self.y_train = None
```

```python
self.y_test = None
```

```python
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

```python
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
```

```python
print("Num TPUs Available: ", len(tf.config.list_physical_devices('TPU')))
```

```python
def load_data(self):
```

```python
sequences = []
```

```python
labels = []
```

```python
max_length = 0
```

*Determine max_length without loading all data*

```python
for folder_path in glob.glob(os.path.join(self.data_path, '*_*')):
```

```python
for subject_file in glob.glob(os.path.join(folder_path, 'sub_*.csv')):
```

```python
with open(subject_file, 'r') as file:
```

```python
row_count = sum(1 for row in file)-1
```

```python
max_length = max(max_length, row_count)
```

*Loop through all files in the data path and load with padding*

```python
for folder_path in glob.glob(os.path.join(self.data_path, '*_*')):
```

```python
action = os.path.basename(folder_path).split('_')[0]
```

```python
for subject_file in glob.glob(os.path.join(folder_path, 'sub_*.csv')):
```

```python
df = pd.read_csv(subject_file)
```

```python
df = df.iloc[:, 1:]
```

```python
df.fillna(0, inplace=True)
```

*Compute statistical features*

```python
df['mean'] = df.mean(axis=1)
```

```python
df['std'] = df.std(axis=1)
```

```python
df['min'] = df.min(axis=1)
```

```python
df['max'] = df.max(axis=1)
```

```python
df['UAx_lag_1'] = df['userAcceleration.x'].shift(1)
```

```python
df['UAx_diff_1'] = df['userAcceleration.x'].diff(1)
```

```python
df['UAy_lag_1'] = df['userAcceleration.y'].shift(1)
```

```python
df['UAy_diff_1'] = df['userAcceleration.y'].diff(1)
```

```python
df['UAz_lag_1'] = df['userAcceleration.z'].shift(1)
```

```python
df['UAz_diff_1'] = df['userAcceleration.z'].diff(1)
```

```python
df['UA_magnitude'] = np.sqrt(np.abs(df['userAcceleration.x']**2 + df['userAcceleration.y']**2 + df['userAcceleration.z']**2))
```

```python
df['Rx_lag_1'] = df['rotationRate.x'].shift(1)
```

```python
df['Rx_diff_1'] = df['rotationRate.x'].diff(1)
```

```python
df['Ry_lag_1'] = df['rotationRate.y'].shift(1)
```

```python
df['Ry_diff_1'] = df['rotationRate.y'].diff(1)
```

```python
df['Rz_lag_1'] = df['rotationRate.z'].shift(1)
```

```python
df['Rz_diff_1'] = df['rotationRate.z'].diff(1)
```

```python
df['R_magnitude'] = np.sqrt(np.abs(df['rotationRate.x']**2 + df['rotationRate.y']**2 + df['rotationRate.z']**2))
```

```python
df['Gx_lag_1'] = df['gravity.x'].shift(1)
```

```python
df['Gx_diff_1'] = df['gravity.x'].diff(1)
```

```python
df['Gy_lag_1'] = df['gravity.y'].shift(1)
```

```python
df['Gy_diff_1'] = df['gravity.y'].diff(1)
```

```python
df['Gz_lag_1'] = df['gravity.z'].shift(1)
```

```python
df['Gz_diff_1'] = df['gravity.z'].diff(1)
```

```python
df['G_magnitude'] = np.sqrt(np.abs(df['gravity.x']**2 + df['gravity.y']**2 + df['gravity.z']**2))
```

```python
df['Ax_lag_1'] = df['attitude.roll'].shift(1)
```

```python
df['Ax_diff_1'] = df['attitude.roll'].diff(1)
```

```python
df['Ay_lag_1'] = df['attitude.pitch'].shift(1)
```

```python
df['Ay_diff_1'] = df['attitude.pitch'].diff(1)
```

```python
df['Az_lag_1'] = df['attitude.yaw'].shift(1)
```

```python
df['Az_diff_1'] = df['attitude.yaw'].diff(1)
```

```python
df['A_magnitude'] = np.sqrt(np.abs(df['attitude.roll']**2 + df['attitude.pitch']**2 + df['attitude.yaw']**2))
```

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
```

```python
df.fillna(0, inplace=True)
```

*Pad the DataFrame with zeros up to max_length rows*

```python
padded_df = pd.DataFrame(index=range(max_length), columns=df.columns).fillna(0)
```

```python
padded_df.iloc[:len(df)] = df.values
```

```python
sequences.append(padded_df.values)
```

```python
labels.append(action)
```

```python
X = np.stack(sequences)
```

```python
y = LabelEncoder().fit_transform(labels)
```

```python
self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
self.X_train = self.X_train.astype('float')
```

```python
self.X_test = self.X_test.astype('float')
```

```python
print(self.X_train.shape, self.X_train.dtype)
```

```python
print(self.y_train.shape, self.y_train.dtype)
```

```python
print(self.X_test.shape, self.X_test.dtype)
```

```python
print(self.y_test.shape, self.y_test.dtype)
```

```python
print("Unique labels in y_train:", np.unique(self.y_train))
```

```python
print("Unique labels in y_test:", np.unique(self.y_test))
```

```python
def preprocess_data(self):
```

```python
self.scaler = MinMaxScaler(feature_range=(0, 1))
```

```python
self.X_train = np.array([self.scaler.fit_transform(x) for x in self.X_train])
```

```python
self.X_test = np.array([self.scaler.transform(x) for x in self.X_test])
```

```python
def train_model(self):
```

```python
self.model = Sequential()
```

```python
self.model.add(Masking(mask_value=0., input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
```

```python
self.model.add(Bidirectional(LSTM(256, return_sequences=True)))
```

```python
self.model.add(Dropout(0.2))
```

```python
self.model.add(Bidirectional(LSTM(256, return_sequences=True)))
```

```python
self.model.add(Dropout(0.2))
```

```python
self.model.add(Bidirectional(LSTM(128, return_sequences=False)))
```

```python
self.model.add(Dense(128, activation='tanh'))
```

```python
self.model.add(Dense(6, activation='softmax'))
```

```python
class_weights = class_weight.compute_class_weight('balanced',classes=
```

```python
np.unique(self.y_train),y=
```

```python
self.y_train)
```

```python
class_weights = dict(enumerate(class_weights))
```

```python
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
```

```python
self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
```

```python
lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
```

```python
self.model.fit(self.X_train, self.y_train, epochs=30, batch_size=16, validation_data=(self.X_test, self.y_test), callbacks=[early_stopping,lr_decay,WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")],class_weight=class_weights)
```

```python
def save_model(self, file_path='HARIoT.h5'):
```

```python
self.model.save(file_path)
```

```python
TC = TrainingClass()
```

```python
TC.load_data()
```

```python
TC.preprocess_data()
```

```python
TC.train_model()
```

```python
TC.save_model()
```

# __pycache__

# assets

# Collision_Prediction

```python
from math import sqrt, sin, cos, radians
```

```python
import matplotlib.pyplot as plt
```

```python
import random
```

```python
from matplotlib.animation import FuncAnimation
```

```python
from math import atan2
```

```python
import pandas as pd
```

```python
import panel as pn
```

```python
import sys
```

```python
import numpy as np
```

The following code is used to send the data to the Orion broker. The data is sent in the form of a JSON file. The data is sent to the topic "Orion_test/contact tracing" on the broker "test.mosquitto.org" The data is sent in the json format but requires the following format:pandas dataframe Adjust the path to the Models accordingly to ensure the modules are used correctly

```python
sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')
```

```python
broker_address="test.mosquitto.org"
```

```python
topic="Orion_test/collision prediction"
```

```python
from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH
```

```python
Handler=MQDH(broker_address, topic)
```

```python
class CustomKalmanFilter:
```

*[Needs Adjustments for complex features]*

```python
def __init__(self, process_variance, measurement_variance, initial_value=(0, 0),
```

```python
initial_velocity=(0, 0), initial_estimate_error=1):
```

```python
self.process_variance = process_variance
```

```python
self.measurement_variance = measurement_variance
```

```python
self.estimate = initial_value
```

```python
self.velocity = initial_velocity
```

```python
self.estimate_error = initial_estimate_error
```

```python
def predict(self, acceleration, direction, dt=1.0):
```

```python
direction_rad = radians(direction)
```

```python
delta_vx = acceleration * cos(direction_rad) * dt
```

```python
delta_vy = acceleration * sin(direction_rad) * dt
```

```python
self.velocity = (self.velocity[0] + delta_vx, self.velocity[1] + delta_vy)
```

```python
self.estimate = (self.estimate[0] + self.velocity[0]*dt + 0.5*delta_vx*dt**2,
```

```python
self.estimate[1] + self.velocity[1]*dt + 0.5*delta_vy*dt**2)
```

```python
self.estimate_error += self.process_variance
```

```python
def update(self, measurement):
```

```python
kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
```

```python
self.estimate = (self.estimate[0] + kalman_gain * (measurement[0] - self.estimate[0]),
```

```python
self.estimate[1] + kalman_gain * (measurement[1] - self.estimate[1]))
```

```python
self.estimate_error *= (1 - kalman_gain)
```

```python
class CollisionPrediction:
```

```python
def __init__(self, process_variance, measurement_variance):
```

```python
self.users = {}
```

```python
self.process_variance = process_variance
```

```python
self.measurement_variance = measurement_variance
```

*Store the latest collision predictions*

```python
self.latest_collisions = []
```

```python
def update_user(self, user_id, coordinates, speed, direction, acceleration):
```

```python
vx = speed * cos(radians(direction))
```

```python
vy = speed * sin(radians(direction))
```

```python
if user_id not in self.users:
```

```python
self.users[user_id] = CustomKalmanFilter(self.process_variance, self.measurement_variance,
```

```python
initial_value=coordinates, initial_velocity=(vx, vy))
```

```python
else:
```

```python
self.users[user_id].predict(acceleration, direction)
```

```python
self.users[user_id].update(coordinates)
```

*Update the latest collision predictions each time user details are updated*

*Using a default prediction_time of 5*

```python
self.latest_collisions = self.predict_collisions(5)
```

```python
def predict_collisions(self, prediction_time, interval=1):
```

Predict collisions at regular intervals within the prediction time.

```python
collisions = set()
```

*Check for collisions at regular intervals*

```python
for t in range(0, prediction_time + 1, interval):
```

```python
user_ids = list(self.users.keys())
```

```python
future_positions = {}
```

```python
for user_id in user_ids:
```

```python
kf = self.users[user_id]
```

```python
future_x = kf.estimate[0] + kf.velocity[0]*t
```

```python
future_y = kf.estimate[1] + kf.velocity[1]*t
```

```python
future_positions[user_id] = (future_x, future_y)
```

```python
for i in range(len(user_ids)):
```

```python
for j in range(i + 1, len(user_ids)):
```

```python
pos1 = future_positions[user_ids[i]]
```

```python
pos2 = future_positions[user_ids[j]]
```

```python
distance = sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```

```python
if distance < 5:
```

```python
collisions.add((user_ids[i], user_ids[j]))
```

```python
return list(collisions)
```

```python
class EnhancedVisualizeMovements:
```

*[Needs Adjustments for complex features]*

```python
def __init__(self, collision_prediction):
```

```python
self.collision_prediction = collision_prediction
```

```python
def compute_distance(self,point1, point2):
```

```python
return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
```

```python
def determine_markersize(self,distance):
```

```python
return 100 / (distance + 1)
```

*Inverse relationship between distance and size*

```python
def compute_intersection(self,initial_position1, velocity1, initial_position2, velocity2):
```

Compute the intersection point of two trajectories given their initial positions and velocities.

*Unpack the initial positions and velocities*

```python
x1, y1 = initial_position1
```

```python
vx1, vy1 = velocity1
```

```python
x2, y2 = initial_position2
```

```python
vx2, vy2 = velocity2
```

*Handle the special cases where the trajectories are vertical*

```python
if vx1 == 0 and vx2 == 0:
```

*Both trajectories are vertical*

```python
return None
```

*No unique intersection point*

```python
elif vx1 == 0:
```

*Only the first trajectory is vertical*

```python
x = x1
```

```python
y = vy2/vx2 * (x - x2) + y2
```

```python
return (x, y)
```

```python
elif vx2 == 0:
```

*Only the second trajectory is vertical*

```python
x = x2
```

```python
y = vy1/vx1 * (x - x1) + y1
```

```python
return (x, y)
```

*Calculate the slopes for the trajectories*

```python
m1 = vy1 / vx1
```

```python
m2 = vy2 / vx2
```

*If the slopes are equal, the trajectories are parallel and do not intersect*

```python
if m1 == m2:
```

```python
return None
```

*Compute x-coordinate of intersection*

```python
x = (y2 - y1 + m1*x1 - m2*x2) / (m1 - m2)
```

*Compute corresponding y-coordinate using one of the trajectory equations*

```python
y = m1 * (x - x1) + y1
```

```python
return (x, y)
```

```python
def bezier_curve(self,ax, start, control, end, **kwargs):
```

```python
t = np.linspace(0, 1, 100)
```

```python
curve = np.outer((1 - t)**2, start) + np.outer(2 * (1 - t) * t, control) + np.outer(t**2, end)
```

```python
ax.plot(curve[:, 0], curve[:, 1], **kwargs)
```

```python
def plot_enhanced_movements(self, ax, prediction_time):
```

Visualize user trajectories and potential collisions.

*Helper function to draw a quadratic Bezier curve*

*Plot initial and predicted positions with intervals*

```python
for user_id, kf in self.collision_prediction.users.items():
```

```python
color = user_colors[user_id]
```

*Use predefined colors for consistency*

```python
initial_x, initial_y = kf.estimate
```

```python
for t in range(0, prediction_time + 1, 1):
```

*Recompute direction at each step*

```python
dx = random.randint(-10, 10)
```

```python
dy = random.randint(-10, 10)
```

```python
predicted_x = initial_x + dx * prediction_time
```

```python
predicted_y = initial_y + dy * prediction_time
```

*Use velocities (if available) or midpoints to determine control point*

```python
control_x = initial_x + dx / 2
```

```python
control_y = initial_y + dy / 2
```

```python
self.bezier_curve(ax, (initial_x, initial_y), (control_x, control_y), (predicted_x, predicted_y), color=color, alpha=0.7)
```

```python
ax.plot(initial_x, initial_y, 's', color=color, markersize=8)
```

```python
ax.annotate('Start', (initial_x, initial_y), textcoords="offset points", xytext=(0,5), ha='center')
```

```python
ax.plot(predicted_x, predicted_y, 'o', color=color, markersize=8)
```

```python
ax.annotate('End', (predicted_x, predicted_y), textcoords="offset points", xytext=(0,5), ha='center')
```

```python
collisions = self.collision_prediction.predict_collisions(prediction_time)
```

```python
for user1, user2 in collisions:
```

```python
future_position1 = self.collision_prediction.users[user1].estimate
```

```python
velocity1 = self.collision_prediction.users[user1].velocity
```

```python
future_position2 = self.collision_prediction.users[user2].estimate
```

```python
velocity2 = self.collision_prediction.users[user2].velocity
```

```python
collision_point = self.compute_intersection(future_position1, velocity1, future_position2, velocity2)
```

```python
if collision_point:
```

*If there's a unique intersection point plot it*

```python
collision_x, collision_y = collision_point
```

```python
distance = self.compute_distance((future_position1[0] + velocity1[0] * prediction_time, future_position1[1] + velocity1[1] * prediction_time),
```

```python
(future_position2[0] + velocity2[0] * prediction_time, future_position2[1] + velocity2[1] * prediction_time))
```

```python
size = self.determine_markersize(distance)
```

```python
ax.plot(collision_x, collision_y, 'ro', markersize=size)
```

```python
try:
```

```python
data={'user1':user1,'user2':user2,'collision_point':collision_point}
```

```python
df=pd.DataFrame(data)
```

```python
if df is not None or isinstance(df, pd.DataFrame):
```

```python
Handler.send_data(df)
```

```python
print("Data Sent to Orion:", df)
```

```python
except:
```

```python
print("Error Sending Data to Orion, current data:",df)
```

```python
continue
```

```python
ax.set_title(f"User Movements: Initial to {prediction_time} Time Units")
```

```python
ax.set_xlabel("X Coordinate")
```

```python
ax.set_ylabel("Y Coordinate")
```

```python
ax.grid(True)
```

```python
ax.legend(loc="upper right")
```

```python
plt.tight_layout()
```

```python
plt.show()
```

*# Send Collision Data to Orion Broker*

*Testing the User class with natural movement*

```python
NUM_USERS = 10
```

```python
NUM_ITERATIONS = 100
```

*Initialize the collision prediction system and visualizer*

```python
collision_prediction = CollisionPrediction(0.1,0.1)
```

```python
visualizer = EnhancedVisualizeMovements(collision_prediction)
```

*Create the initial plot*

```python
fig, ax = plt.subplots()
```

```python
ax.set_xlim(-500, 500)
```

```python
ax.set_ylim(-500, 500)
```

```python
ax.set_title("User Movements Animation")
```

```python
ax.set_xlabel("X Coordinate")
```

```python
ax.set_ylabel("Y Coordinate")
```

*Initialize user_positions dictionary outside the update function to persist positions across frames*

```python
user_positions = {}
```

```python
def adjusted_color_map(index, total):
```

*Adjust the index to skip the red portion of the jet colormap*

*This can be fine-tuned based on the exact portion of the colormap you want to exclude*

```python
if index / total > 0.7:
```

```python
index += int(0.2 * total)
```

*skip 20% of the colormap after the 70% mark*

```python
return plt.cm.jet(index / total)
```

*Assigning a fixed color to each user to ensure consistent coloring across frames*

```python
user_colors = {user_id: adjusted_color_map(i, NUM_USERS) for i, user_id in enumerate(range(1, NUM_USERS + 1))}
```

```python
def update(frame):
```

```python
ax.clear()
```

```python
ax.set_xlim(-100, 100)
```

*Adjusting limits to match the earlier defined space*

```python
ax.set_ylim(-100, 100)
```

```python
for user_id in range(1, NUM_USERS + 1):
```

*Check the current position of the user*

```python
current_position = user_positions.get(user_id, (random.randint(0, 100), random.randint(0, 100)))
```

*Generate incremental movement parameters*

```python
dx = random.randint(-1, 1)
```

```python
dy = random.randint(-1, 1)
```

```python
x = current_position[0] + dx
```

```python
y = current_position[1] + dy
```

```python
user_positions[user_id] = (x, y)
```

*Update the new position in the dictionary*

```python
speed = (dx**2 + dy**2)**0.5
```

```python
direction = (180 / 3.14159) * atan2(dy, dx)
```

```python
direction = (180 / 3.14159) * atan2(dy, dx)
```

*Introduce a random direction shift*

```python
angle_shift = random.uniform(-90, 90)
```

*Random shift between -30 and 30 degrees*

```python
direction += angle_shift
```

```python
acceleration = 0
```

*Assuming no acceleration for simplicity*

*Update user's movement in the collision prediction system*

```python
collision_prediction.update_user(user_id, (x, y), speed, direction, acceleration)
```

*Plot the user's position with a fixed color*

```python
ax.plot(x, y, 'o', color=user_colors[user_id],label=f"User {user_id}")
```

*Visualize the movements and potential collisions*

```python
visualizer.plot_enhanced_movements(ax, 10)
```

```python
plot_panel = pn.pane.Matplotlib(fig, tight=True)
```

*Create a Panel pane from the Matplotlib figure*

*Create the animation*

```python
ani = FuncAnimation(fig, update, frames=range(NUM_ITERATIONS), repeat=False)
```

```python
plt.show()
```

*Empty to allow this folder to be treated as a package and allow communication between files*

# __pycache__

# Contact_Tracing

```python
import pandas as pd
```

```python
from datetime import datetime,timedelta
```

```python
import matplotlib.pyplot as plt
```

```python
import numpy as np
```

```python
import json
```

```python
import sys
```

```python
import panel as pn
```

The following code before the class is used to send the data to the Orion broker. The data is sent in the form of a JSON file. The data is sent to the topic "Orion_test/contact tracing" on the broker "test.mosquitto.org" The data is sent in the json format but requires the following format:pandas dataframe Adjust the path to the Models accordingly to ensure the modules are used correctly

```python
sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')
```

```python
broker_address="test.mosquitto.org"
```

```python
topic="Orion_test/contact tracing"
```

```python
from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH
```

```python
Handler=MQDH(broker_address, topic)
```

The class below is used for contact tracing it requires the following inputs: UserID: The user ID of the user Coordinates: The coordinates of the user Timestamp: The timestamp of the user at the coordinates  The class has the following functions: add_record: Adds a new record to the dataframe get_time_based_contacts: Gets the contacts of the user based on the time window and radius   It can be extended to include more functions as required for example a function to get the contacts of the user based on the location or a neural network to predict the contacts of the user based on the location and time

```python
class ContactTracer:
```

```python
def __init__(self):
```

```python
self.data = pd.DataFrame(columns=["UserID", "Coordinates", "Timestamp"])
```

```python
def add_record(self, user_id, coordinates, timestamp=None):
```

Add a new location record for a user using pandas.concat.""" if timestamp is None: timestamp = datetime.now() new_record = pd.DataFrame({"UserID": [user_id], "Coordinates": [coordinates], "Timestamp": [timestamp]}) self.data = pd.concat([self.data, new_record], ignore_index=True)  def get_time_based_contacts(self, user_id, radius, time_window=timedelta(minutes=30)):

```python
user_data = self.data[self.data["UserID"] == user_id]
```

```python
potential_contacts = pd.DataFrame()
```

```python
for _, record in user_data.iterrows():
```

```python
lat1, lon1 = record["Coordinates"]
```

```python
timestamp = record["Timestamp"]
```

```python
contacts = self.data[
```

```python
(self.data["Timestamp"] - timestamp).abs() <= time_window
```

*time condition*

```python
]
```

```python
for _, contact in contacts.iterrows():
```

```python
lat2, lon2 = contact["Coordinates"]
```

```python
distance = ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5
```

*simple distance formula*

```python
if distance <= radius and contact["UserID"] != user_id:
```

```python
potential_contacts = pd.concat([potential_contacts, contact.to_frame().T], ignore_index=True)
```

*Drop duplicate rows*

```python
potential_contacts = potential_contacts.drop_duplicates()
```

```python
return potential_contacts
```

*Re-populate the ContactTracer instance with the previously generated data (with timestamps)*

*Repopulating the ContactTracer instance with the provided data and adding timestamps*

*Resetting the ContactTracer instance*

```python
tracer = ContactTracer()
```

*Adding records with timestamps*

```python
records = [
```

```python
("UserA", (1, 2)),
```

```python
("UserB", (2, 2)),
```

```python
("UserC", (10, 10)),
```

```python
("UserA", (3, 2)),
```

```python
("UserA", (4, 3)),
```

```python
("UserA", (4, 4)),
```

```python
("UserD", (4, 5)),
```

```python
("UserD", (5, 5)),
```

```python
("UserD", (4, 7)),
```

```python
("UserD", (4, 8)),
```

```python
("UserD", (5, 9)),
```

```python
("UserD", (6, 10)),
```

```python
("UserD", (8, 11)),
```

```python
("UserE", (9, 12)),
```

```python
("UserE", (10, 12))
```

```python
]
```

*Assigning a unique timestamp for each record*

```python
base_timestamp = datetime.now()
```

```python
for i, (user, coords) in enumerate(records):
```

```python
timestamp = base_timestamp + timedelta(minutes=i)
```

*Each record is 1 minute apart*

```python
tracer.add_record(user, coords, timestamp)
```

*getting all user contacts to send to the Orion broker*

```python
for user in tracer.data["UserID"].unique():
```

```python
contacts = tracer.get_time_based_contacts(user, 2)
```

```python
print(f"Contacts of {user}:")
```

```python
print(contacts)
```

```python
df=pd.DataFrame(contacts)
```

*Handler.send_data(df, user_id=user)#uncomment to send data to the broker*

*Getting contacts of UserA within a radius of 2 units*

```python
contacts = tracer.get_time_based_contacts("UserA", 2)
```

```python
def plot_spatial_temporal_contacts(central_user, contacts_df):
```

```python
figure=plt.figure(figsize=(12, 10))
```

*Plot the central user at (0,0) for simplicity*

```python
plt.scatter(0, 0, color="red", label=central_user, s=200, zorder=5)
```

```python
plt.text(0, 0, central_user, fontsize=12, ha='right')
```

*Plot the contacts*

```python
colors = plt.cm.rainbow(np.linspace(0, 1, len(contacts_df["UserID"].unique())))
```

```python
color_map = dict(zip(contacts_df["UserID"].unique(), colors))
```

```python
for _, row in contacts_df.iterrows():
```

```python
x, y = row["Coordinates"]
```

```python
user = row["UserID"]
```

```python
plt.scatter(x, y, color=color_map[user], s=100)
```

```python
plt.text(x, y, user, fontsize=12, ha='right')
```

*Draw a line between the central user and the contact*

```python
plt.plot([0, x], [0, y], color=color_map[user], linestyle='--', linewidth=1)
```

*Annotate the line with the timestamp of contact*

```python
midpoint = ((x+0)/2, (y+0)/2)
```

```python
plt.annotate(row["Timestamp"].strftime('%H:%M:%S'),
```

```python
xy=midpoint,
```

```python
xytext=midpoint,
```

```python
fontsize=10,
```

```python
arrowprops=dict(facecolor='black', arrowstyle='-'),
```

```python
ha='center')
```

```python
plt.xlabel("Longitude")
```

```python
plt.ylabel("Latitude")
```

```python
plt.title(f"Spatial-Temporal Contacts of {central_user}")
```

```python
plt.grid(True)
```

```python
return figure
```

*Plotting the contacts of "UserA"*

```python
fig=plot_spatial_temporal_contacts("UserA", contacts)
```

```python
plt.close(fig)
```

```python
plot_panel = pn.pane.Matplotlib(fig, tight=True)
```

*Empty to allow this folder to be treated as a package and allow communication between files*

# __pycache__

# Dashboard

```python
import panel as pn
```

```python
import os
```

```python
import sys
```

```python
import matplotlib.pyplot as plt
```

This is the Dashboard class that will be used to create the dashboard. It has the following functions: add_plot: Adds a plot to the dashboard add_widget: Adds a widget to the dashboard add_detail: Adds a detail to the dashboard construct_dashboard: Constructs the dashboard show: Displays the dashboard   The dashboard class can be used in conjuction with other modules to create a dashboard. For example, the dashboard can be used with the MQTTManager to display the data received from the MQTT broker. The dashboard can also be used with the ContactTracer to display the contacts of the user based on the time and location

```python
sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')
```

```python
from Collision_Prediction import Co_Pred
```

```python
from Contact_Tracing import Tracer
```

```python
from HeartRate_Monitoring import HeartRateMonitor
```

*from Individual_Tracking import Tracker_run*

```python
from Overcrowding_Detection import ODHG
```

```python
Collision_Panel = Co_Pred.plot_panel
```

```python
Contact_Panel = Tracer.plot_panel
```

```python
HeartRate_Panel = HeartRateMonitor.plot_panel
```

*Individual_Panel = Tracker_run.plot_panel()*

```python
Overcrowding_Panel = ODHG.plot_panel
```

```python
class Dashboard:
```

```python
def __init__(self, plots=None, widgets=None, details=None):
```

```python
self.plots = plots if plots else []
```

```python
self.widgets = widgets if widgets else []
```

```python
self.details = details if details else []
```

```python
def save_plot_as_image(self, plot, img_name):
```

Save the provided Matplotlib figure or Panel Matplotlib pane as an image and return a PNG pane.""" # Check if the plot is a Panel Matplotlib pane and extract the figure if so if isinstance(plot, pn.pane.Matplotlib): fig = plot.object elif isinstance(plot, plt.Axes):  # Check if it's an AxesSubplot object fig = plot.figure else: fig = plot  # Assuming you save images in an 'images' directory images=r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Dashboard\images' images_dir = os.path.join(images, img_name) fig.savefig(images_dir) image_pane = pn.pane.PNG(images_dir, width=500) return image_pane def add_plot(self, fig, img_name):

```python
image_pane = self.save_plot_as_image(fig, img_name)
```

```python
self.plots.append(image_pane)
```

```python
def add_gif_to_dashboard(self, gif_path):
```

Add a GIF to the Panel dashboard.  Parameters: - gif_path (str): Path to the GIF file.

```python
gif_pane = pn.pane.Image(gif_path, width=500)
```

```python
self.plots.append(gif_pane)
```

```python
def add_widget(self, widget):
```

Add a widget to the dashboard.""" if widget: self.widgets.append(widget)  def add_detail(self, detail):

```python
if detail:
```

```python
self.details.append(detail)
```

```python
def construct_dashboard(self):
```

Construct and return the dashboard layout by interleaving plots and details.""" items = []  # Adding widgets if any if self.widgets: items.append(pn.Row(*self.widgets))  # Interleave plots and details for plot, detail in zip(self.plots, self.details): items.append(plot) items.append(detail)  # If there are any remaining plots or details, add them as well remaining_plots = len(self.plots) - len(self.details) for i in range(remaining_plots): items.append(self.plots[len(self.details) + i])  remaining_details = len(self.details) - len(self.plots) for i in range(remaining_details): items.append(self.details[len(self.plots) + i])  dashboard = pn.Column(*items) return dashboard  def show(self):

```python
dashboard = self.construct_dashboard()
```

```python
dashboard.show()
```

*Create the Dashboard instance and add plots*

```python
dashboard_instance = Dashboard()
```

```python
dashboard_instance.add_gif_to_dashboard(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Dashboard\images\Collision_P.gif')
```

```python
coll_deets='The above plot provides insights into potential collisions among tracked users over a specified timeframe. Each colored line represents the predicted trajectory of a different user, with the starting point marked as -Start- and the predicted endpoint as -End-. The presence of a red circle indicates a predicted point of collision. The size of the circle is proportional to the likelihood of a collision, with larger circles signifying higher probabilities. Such predictions can be instrumental in ensuring user safety by preempting and averting possible collisions.'
```

```python
dashboard_instance.add_detail(coll_deets)
```

```python
dashboard_instance.add_plot(Tracer.plot_panel, "Contact_Tracing.png")
```

```python
cont_deets='The plot above is a visual representation of user contacts based on temporal and spatial data. It traces the interactions between different users, highlighting moments of close proximity. Such visualizations are crucial, especially in scenarios like infectious disease outbreaks, where understanding person-to-person contact patterns can play a pivotal role in containment and mitigation strategies.'
```

```python
dashboard_instance.add_detail(cont_deets)
```

```python
dashboard_instance.add_plot(HeartRateMonitor.plot_panel, "HeartRate_Monitoring.png")
```

```python
HR_deets='The displayed plot captures the heart rate data of users over time, providing a continuous monitor of their cardiovascular health. Fluctuations, spikes, or irregular patterns can be indicative of underlying health conditions or moments of heightened stress or activity. Regular monitoring can aid in timely interventions and ensure the well-being of users.'
```

```python
dashboard_instance.add_detail(HR_deets)
```

*dashboard_instance.add_plot(Tracker_run.plot_panel)*

```python
dashboard_instance.add_plot(ODHG.plot_panel, "Overcrowding_Detection.png")
```

```python
dashboard_instance.add_plot(ODHG.plot_panel2, "Overcrowding_Detection2.png")
```

```python
OD_deets='The presented visualizations delve into crowd density estimation within a specified area. The first plot delineates clusters, showcasing the congregation of individuals in specific zones. The subsequent heatmap offers a gradient view of crowd density, with warmer colors signifying areas of higher congestion. Such depictions are invaluable in scenarios that demand crowd management, ensuring safety protocols, and optimizing space utilization.'
```

```python
dashboard_instance.add_detail(OD_deets)
```

*Construct and save the dashboard*

```python
constructed_dashboard = dashboard_instance.construct_dashboard()
```

```python
constructed_dashboard.save(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Dashboard\dashboard.html')
```

# images

# __pycache__

# DataManager

```python
import pandas as pd
```

```python
import paho.mqtt.client as mqtt
```

```python
import json
```

```python
import time
```

```python
import ssl
```

*Importing ssl certificates - ensures data transmitted is encrypted*

```python
import datetime
```

```python
from datetime import datetime
```

```python
from cryptography.fernet import Fernet
```

*importing cryptogtaphy library*

The class below is used to send and receive data from the MQTT broker. The data is sent in the form of a JSON file. The data is sent to the topic "test/topic" on the broker "test.mosquitto.org"/any other broker

*adding an encryption and decryption key*

```python
encryption_key = Fernet.generate_key()
```

```python
cipher_suite = Fernet(encryption_key)
```

```python
class MQTTDataFrameHandler:
```

```python
def __init__(self, broker_address, topic, max_retries=3, retry_interval=5):
```

```python
self.broker_address = broker_address
```

```python
self.topic = topic
```

```python
self.client = mqtt.Client()
```

```python
self.client.on_message = self._on_message
```

```python
self.data = None
```

```python
self.error = None
```

```python
self.max_retries = max_retries
```

```python
self.retry_interval = retry_interval
```

```python
def _on_message(self, client, userdata, message):
```

```python
try:
```

*Convert received message payload to DataFrame*

```python
encrypted_data = message.payload
```

*decrypt and convert received message*

```python
data_json = cipher_suite.decrypt(encrypted_data).decode('utf-8')
```

*uses key to decrypt data*

*Convert received message payload to DataFrame*

```python
data_json = message.payload.decode('utf-8')
```

```python
self.data = pd.read_json(data_json)
```

*Add a timestamp column to the DataFrame to allow tracking of data age*

```python
self.data['timestamp'] = time.time()
```

```python
except Exception as e:
```

```python
self.error = str(e)
```

```python
def create_json_payload(self, dataframe, user_id=None):
```

*Convert dataframe to JSON format*

```python
data_json = dataframe.to_json(orient='split')
```

```python
payload = {
```

```python
'timestamp': datetime.utcnow().isoformat(),
```

```python
'data': json.loads(data_json)
```

```python
}
```

```python
if user_id:
```

```python
payload['user_id'] = user_id
```

```python
return json.dumps(payload)
```

```python
def receive_data(self, timeout=10):
```

```python
retries = 0
```

```python
while retries < self.max_retries:
```

```python
try:
```

```python
self.client.connect(self.broker_address, 1883, 60)
```

```python
self.client.subscribe(self.topic)
```

```python
self.client.loop_start()
```

```python
start_time = time.time()
```

```python
while self.data is None and (time.time() - start_time) < timeout:
```

```python
if self.error:
```

```python
print(f"Error while receiving data: {self.error}")
```

```python
break
```

```python
self.client.loop_stop()
```

```python
return self.data
```

```python
except Exception as e:
```

```python
print(f"Connection error: {e}. Retrying in {self.retry_interval} seconds...")
```

```python
retries += 1
```

```python
time.sleep(self.retry_interval)
```

```python
print("Max retries reached. Failed to receive data.")
```

```python
return None
```

```python
def send_data(self, df , user_id=None):
```

```python
retries = 0
```

```python
while retries < self.max_retries:
```

```python
try:
```

```python
json_payload = self.create_json_payload(df,user_id)
```

```python
self.client.connect(self.broker_address, 1883, 60)
```

```python
self.client.publish(self.topic, json_payload)
```

```python
self.client.disconnect()
```

```python
return
```

```python
except Exception as e:
```

```python
print(f"Error while sending data: {e}. Retrying in {self.retry_interval} seconds...")
```

```python
retries += 1
```

```python
time.sleep(self.retry_interval)
```

```python
print("Max retries reached. Failed to send data.")
```

```python
def main():
```

*placeholders*

```python
broker_address = "test.mosquitto.org"
```

```python
topic = "test/topic"
```

```python
handler = MQTTDataFrameHandler(broker_address, topic)
```

*SSL setup for client*

```python
handler.client.tls_set(ca_certs="ca.crt", certfile="client.crt", keyfile="client.key", tls_version=ssl.PROTOCOL_TLS)
```

*Path should be rewritten to find actual files*

```python
handler.client.username_pw_set("client_username", "client_password")
```

*'client_username' and 'cliet_password' should be replaced with client's actual username and password*

```python
if __name__ == "__main__":
```

```python
main()
```

```python
from flask import Flask, request
```

```python
app = Flask(__name__)
```

```python
@app.route('/receive_data', methods=['POST'])
```

```python
def receive_data():
```

```python
data = request.json
```

```python
handler = DataHandler()
```

```python
handler.save_data(data)
```

```python
print("data saved")
```

```python
return "Data received", 200
```

The class below is used in conjuction with the provided unity virtual environment  to recieve virtual data in realtime to simulate a live service. The data recieved is in a static format so any changes made here should be reflected in the code within unity  This class act as a server to recieve the data, this data can then be used to populate the database and evaluate the models/modules, this data can also be used to populate the dashboard  The data is completely random and is not based on any real world data and also with every run the unity environment creates a new set of data but the format remains the same, the current setting is San Francisco, USA for gps data

```python
if __name__ == '__main__':
```

```python
app.run(port=5000)
```

```python
import sqlite3
```

```python
class DataHandler:
```

```python
def __init__(self):
```

```python
self.conn = sqlite3.connect('data.db')
```

```python
self.create_table()
```

```python
def create_table(self):
```

```python
with self.conn:
```

```python
self.conn.execute('''
```

```python
CREATE TABLE IF NOT EXISTS user_data (
```

```python
Time TEXT,
```

```python
Agent_ID TEXT,
```

```python
Device_Type TEXT,
```

```python
Accelerometer_X TEXT,
```

```python
Accelerometer_Y TEXT,
```

```python
Accelerometer_Z TEXT,
```

```python
Gyroscope_X TEXT,
```

```python
Gyroscope_Y TEXT,
```

```python
Gyroscope_Z TEXT,
```

```python
Longitude_Degrees TEXT,
```

```python
Longitude_Minutes TEXT,
```

```python
Latitude_Degrees TEXT,
```

```python
Latitude_Minutes TEXT,
```

```python
Altitude TEXT
```

```python
)
```

)  def save_data(self, data): with self.conn: self.conn.execute(''' INSERT INTO user_data ( Time, Agent_ID, Device_Type, Accelerometer_X, Accelerometer_Y, Accelerometer_Z, Gyroscope_X, Gyroscope_Y, Gyroscope_Z, Longitude_Degrees, Longitude_Minutes, Latitude_Degrees, Latitude_Minutes, Altitude ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

```python
data['Time'], data['Agent ID'], data['Device Type'],
```

```python
data['Accelerometer X'], data['Accelerometer Y'], data['Accelerometer Z'],
```

```python
data['Gyroscope X'], data['Gyroscope Y'], data['Gyroscope Z'],
```

```python
data['Longitude Degrees'], data['Longitude Minutes'],
```

```python
data['Latitude Degrees'], data['Latitude Minutes'], data['Altitude']
```

```python
))
```

```python
def get_all_data_for_user(self, user_id, minimum_entries=500):
```

```python
with self.conn:
```

```python
cursor = self.conn.execute('SELECT * FROM user_data WHERE Agent_ID = ?', (user_id,))
```

```python
data = cursor.fetchall()
```

```python
if len(data) < minimum_entries:
```

```python
return None
```

*or return an empty list*

```python
return data
```

```python
def get_latest_data_for_user(self, user_id):
```

```python
with self.conn:
```

```python
cursor = self.conn.execute('SELECT * FROM user_data WHERE Agent_ID = ? ORDER BY Time DESC LIMIT 1', (user_id,))
```

```python
return cursor.fetchone()
```

```python
def get_all_user_coordinates(self,user_id):
```

```python
with self.conn:
```

```python
cursor = self.conn.execute('SELECT Time, Altitude, Latitude_Degrees, Longitude_Degrees FROM user_data WHERE Agent_ID = ?', (user_id,))
```

```python
return cursor.fetchall()
```

```python
def get_all_orentation_data_for_user(self, user_id):
```

```python
with self.conn:
```

```python
cursor = self.conn.execute('SELECT Time,Acceleremoter_X,Acceleremoter_Y,Acceleremoter_Z,Gyroscope_X,Gyroscope_Y,Gyroscope_Z FROM user_data WHERE Agent_ID = ?', (user_id,))
```

```python
data = cursor.fetchall()
```

```python
return data
```

*Empty to allow this folder to be treated as a package and allow communication between files*

# __pycache__

# HeartRate_Monitoring

```python
from datetime import datetime, timedelta
```

```python
import pandas as pd
```

```python
import numpy as np
```

```python
from scipy.signal import find_peaks
```

```python
import random
```

```python
def generate_ppg_data(number_of_individuals, sample_frequency, duration):
```

Generate synthetic PPG data for simulating both healthy and unhealthy individuals.  **Parameter:** number_of_individuals: int, the number of individual PPG data sets to create **Parameter:** sample_frequency: float, the sampling frequency in Hz **Parameter:** duration: float, the total duration in seconds for each individual's PPG data

*Constants for mean heart rates of healthy and unhealthy conditions*

```python
healthy_mean_hr = 75
```

```python
unhealthy_mean_hr = 90
```

```python
noise_level = 0.05
```

*Noise level for the generated signal*

```python
for individual in range(number_of_individuals):
```

*Randomly determine the start date and time for the simulation*

```python
start_datetime = datetime(random.randint(2019,2023), random.randint(1,12), random.randint(1,28), random.randint(0,23), random.randint(0,59), random.randint(0,59))
```

*Prepare the CSV data*

```python
csv_data = []
```

```python
columns = ['DayOfWeek', 'Hour', 'Minute', 'Second', 'Condition', 'PPG_Signal']
```

```python
condition = 'Healthy' if np.random.rand() > 0.2 else 'Unhealthy'
```

*Assign condition with 20% probability of being unhealthy*

```python
mean_heart_rate = healthy_mean_hr if condition == 'Healthy' else unhealthy_mean_hr
```

*Generate time intervals*

```python
time_intervals = [start_datetime + timedelta(seconds=t) for t in np.arange(0, duration, 1 / sample_frequency)]
```

*Generate baseline and heartbeat signals*

```python
baseline_signal = np.sin(2 * np.pi * mean_heart_rate / 60 * np.arange(0, duration, 1 / sample_frequency))
```

```python
heartbeat_signal = np.sin(2 * np.pi * mean_heart_rate / 60 * 10 * np.arange(0, duration, 1 / sample_frequency))
```

```python
peaks, _ = find_peaks(heartbeat_signal, height=0.5)
```

```python
heartbeat_signal *= 0
```

```python
heartbeat_signal[peaks] = 1
```

*Combine signals and add noise*

```python
ppg_signal = baseline_signal + heartbeat_signal
```

```python
ppg_signal += np.random.normal(0, noise_level, len(ppg_signal))
```

*Append to CSV data with DateTime format*

```python
for datetime_interval, ppg in zip(time_intervals, ppg_signal):
```

```python
day_of_week = datetime_interval.isoweekday()
```

*Monday is 1, Sunday is 7*

```python
hour = datetime_interval.hour
```

```python
minute = datetime_interval.minute
```

```python
second = datetime_interval.second
```

```python
csv_data.append([day_of_week, hour, minute, second, condition, ppg])
```

*Create a DataFrame*

```python
df = pd.DataFrame(csv_data, columns=columns)
```

*Save to CSV file*

```python
file_name = f"E:\\Dev\\Deakin\\Project_Orion\\DataScience\\Models\\HeartRate Monitoring\\HRD\\individual_{individual}.csv"
```

```python
df.to_csv(file_name, index=False)
```

*Example usage to generate individual CSV files*

```python
generate_ppg_data(number_of_individuals=100, sample_frequency=100, duration=600)
```

```python
import math
```

```python
import numpy as np
```

```python
import datetime
```

```python
import pandas as pd
```

```python
import numpy as np
```

```python
import matplotlib.pyplot as plt
```

```python
import os
```

```python
import panel as pn
```

```python
from scipy.fft import fft
```

```python
from scipy.stats import skew, kurtosis, zscore
```

```python
from sklearn.preprocessing import StandardScaler
```

```python
from sklearn.model_selection import train_test_split
```

```python
from sklearn.preprocessing import LabelEncoder
```

```python
from sklearn.linear_model import LinearRegression
```

```python
from sklearn.ensemble import RandomForestRegressor
```

```python
from sklearn.metrics import mean_squared_error
```

```python
from sklearn.svm import OneClassSVM
```

The HRMonitoring class is meant to manage the data input and output of the system. It is the main class that calls the other classes and functions.

```python
class HRmonitoring():
```

```python
def __init__(self,userID):
```

```python
self.userID = userID
```

```python
self.data = None
```

```python
self.features = None
```

```python
def get_data(self, file_path):
```

```python
self.data = pd.read_csv(file_path)
```

```python
self.data['userID'] = self.userID
```

```python
def create_sequences(data, window_size):
```

```python
sequences = []
```

```python
for i in range(len(data) - window_size + 1):
```

```python
sequences.append(data[i:i+window_size])
```

```python
return np.array(sequences)
```

```python
def extract_features(self):
```

*Ensure data is present*

```python
if self.data is None:
```

```python
print("Data not loaded.")
```

```python
return
```

*Baseline Heart Rate*

```python
baseline_hr = np.mean(self.data['PPG_Signal'])
```

*Heart Rate Variability (HRV)*

```python
rr_intervals = np.diff(self.data['PPG_Signal'].values)
```

*Difference between successive heart rates*

```python
hrv = np.std(rr_intervals)
```

*Outliers using IQR*

```python
Q1 = np.percentile(self.data['PPG_Signal'], 25)
```

```python
Q3 = np.percentile(self.data['PPG_Signal'], 75)
```

```python
IQR = Q3 - Q1
```

```python
outlier_count = np.sum((self.data['PPG_Signal'] < (Q1 - 1.5 * IQR)) | (self.data['PPG_Signal'] > (Q3 + 1.5 * IQR)))
```

*Frequency Domain Features using Fourier Transform*

```python
yf = fft(self.data['PPG_Signal'].to_numpy())
```

```python
power_spectrum = np.abs(yf)
```

```python
dominant_frequency = np.argmax(power_spectrum)
```

*Moving Average (7-day window as an example)*

```python
moving_avg = self.data['PPG_Signal'].rolling(window=7).mean()
```

*Skewness and Kurtosis*

```python
skewness = skew(self.data['PPG_Signal'])
```

```python
kurt = kurtosis(self.data['PPG_Signal'])
```

*Store features*

```python
self.features = {
```

```python
'baseline_hr': baseline_hr,
```

```python
'hrv': hrv,
```

```python
'outlier_count': outlier_count,
```

```python
'dominant_frequency': dominant_frequency,
```

```python
'moving_avg': moving_avg,
```

```python
'skewness': skewness,
```

```python
'kurtosis': kurt
```

```python
}
```

```python
def detect_spikes(self, window_size=5, threshold=2.0):
```

```python
if self.data is None:
```

```python
print("Data not loaded.")
```

```python
return
```

*Calculate rolling mean and standard deviation*

```python
rolling_mean = self.data['PPG_Signal'].rolling(window=window_size).mean()
```

```python
rolling_std = self.data['PPG_Signal'].rolling(window=window_size).std()
```

*Detect spikes where the signal deviates from the rolling mean by more than the threshold times the rolling standard deviation*

```python
spikes = np.where(np.abs(self.data['PPG_Signal'] - rolling_mean) > threshold * rolling_std)[0]
```

```python
return spikes
```

```python
def detect_shifts(self, window_size=50, threshold=2.0):
```

```python
if self.data is None:
```

```python
print("Data not loaded.")
```

```python
return
```

*Calculate rolling mean for two consecutive windows*

```python
rolling_mean_1 = self.data['PPG_Signal'].rolling(window=window_size).mean().shift(-window_size)
```

```python
rolling_mean_2 = self.data['PPG_Signal'].rolling(window=window_size).mean()
```

*Detect shifts where the difference between the means of two consecutive windows is greater than the threshold*

```python
shifts = np.where(np.abs(rolling_mean_1 - rolling_mean_2) > threshold)[0]
```

```python
return shifts
```

```python
def process():
```

*Encoding the 'Condition' column*

```python
label_encoder = LabelEncoder()
```

```python
df['Condition_encoded'] = label_encoder.fit_transform(df['Condition'])
```

*Drop the original 'Condition' column and use the encoded one*

```python
df = df.drop(columns=['Condition'])
```

*Splitting the data into training and testing sets*

```python
X = df.drop(columns=['PPG_Signal'])
```

```python
y = df['PPG_Signal']
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
X_train.head()
```

```python
return X_train, X_test, y_train, y_test
```

```python
def train(X_train, X_test, y_train, y_test):
```

*Training Random Forest Regressor*

```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
```

```python
rf_model.fit(X_train, y_train)
```

*Predicting on the test set*

```python
rf_predictions = rf_model.predict(X_test)
```

*Evaluate the model using Mean Squared Error (MSE)*

```python
rf_mse = mean_squared_error(y_test, rf_predictions)
```

```python
return rf_model, rf_mse
```

```python
def predict(model, X_test):
```

*Predicting on the test set*

```python
predictions = model.predict(X_test)
```

```python
return predictions
```

```python
def train_anomaly_detector(self, sample_size=10000):
```

*Sample a subset of the data for training the One-Class SVM*

```python
subset_data = self.data.sample(sample_size, random_state=42)
```

*Prepare features for the model*

```python
features = subset_data[['PPG_Signal']]
```

*Train the One-Class SVM*

```python
self.anomaly_model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
```

```python
self.anomaly_model.fit(features)
```

```python
def detect_anomalies(self):
```

```python
if self.anomaly_model is None:
```

```python
print("Model not trained. Use the train_anomaly_detector method first.")
```

```python
return
```

*Predict anomalies on the data*

```python
anomalies = self.anomaly_model.predict(self.data[['PPG_Signal']])
```

*Filter out the anomaly rows*

```python
anomaly_data = self.data[anomalies == -1]
```

```python
return anomaly_data
```

```python
path=r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\HeartRate_Monitoring\HRD'
```

```python
monitor = HRmonitoring(userID="individual_0")
```

```python
monitor.get_data(os.path.join(path, "individual_0.csv"))
```

*Extract features using the class methods*

```python
monitor.extract_features()
```

*Detect spikes and shifts*

```python
spikes = monitor.detect_spikes()
```

```python
shifts = monitor.detect_shifts()
```

*Train the anomaly detector and detect anomalies*

```python
monitor.train_anomaly_detector()
```

```python
anomalies = monitor.detect_anomalies()
```

*Plot the results*

```python
import matplotlib.pyplot as plt
```

```python
plt.figure(figsize=(15, 8))
```

```python
plt.plot(monitor.data['PPG_Signal'], label="PPG Signal", alpha=0.7)
```

```python
plt.scatter(spikes, monitor.data['PPG_Signal'].iloc[spikes], color='red', label="Detected Spikes", s=50, marker="x")
```

```python
plt.scatter(shifts, monitor.data['PPG_Signal'].iloc[shifts], color='blue', label="Detected Shifts", s=50, marker="o")
```

```python
plt.scatter(anomalies.index, anomalies['PPG_Signal'], color='green', label="Detected Anomalies", s=50, marker="s")
```

```python
plt.legend()
```

```python
plt.title("PPG Signal with Detected Spikes, Shifts, and Anomalies")
```

```python
plt.xlabel("Time (Samples)")
```

```python
plt.ylabel("PPG Signal Value")
```

```python
plt.tight_layout()
```

```python
plt.show()
```

```python
monitor.extract_features()
```

*Plotting the extracted features*

```python
fig, axs = plt.subplots(4, 1, figsize=(15, 15))
```

*PPG Signal with Moving Average and Outliers*

```python
axs[0].plot(monitor.data['PPG_Signal'], label="PPG Signal", alpha=0.7)
```

```python
axs[0].plot(monitor.features['moving_avg'], label="Moving Average (7-sample window)", color="purple")
```

```python
Q1 = np.percentile(monitor.data['PPG_Signal'], 25)
```

```python
Q3 = np.percentile(monitor.data['PPG_Signal'], 75)
```

```python
IQR = Q3 - Q1
```

```python
outliers = monitor.data[(monitor.data['PPG_Signal'] < (Q1 - 1.5 * IQR)) | (monitor.data['PPG_Signal'] > (Q3 + 1.5 * IQR))]
```

```python
axs[0].scatter(outliers.index, outliers['PPG_Signal'], color='red', label="IQR Outliers", s=50, marker="x")
```

```python
axs[0].set_title("PPG Signal with Moving Average and IQR Outliers")
```

```python
axs[0].set_xlabel("Time (Samples)")
```

```python
axs[0].set_ylabel("PPG Signal Value")
```

```python
axs[0].legend()
```

*Frequency Domain Representation of PPG Signal*

```python
yf = fft(monitor.data['PPG_Signal'].to_numpy())
```

```python
power_spectrum = np.abs(yf[:len(yf)//2])
```

*Taking half due to symmetry*

```python
frequencies = np.linspace(0, 0.5, len(power_spectrum))
```

*Frequency values from 0 to Nyquist (0.5 for normalized freq.)*

```python
axs[1].plot(frequencies, power_spectrum, label="Frequency Spectrum", alpha=0.7)
```

```python
axs[1].axvline(frequencies[monitor.features['dominant_frequency']], color='red', linestyle='--', label="Dominant Frequency")
```

```python
axs[1].set_title("Frequency Domain Representation of PPG Signal")
```

```python
axs[1].set_xlabel("Frequency (Normalized)")
```

```python
axs[1].set_ylabel("Magnitude")
```

```python
axs[1].legend()
```

*Displaying HRV*

```python
axs[2].text(0.5, 0.6, f"Heart Rate Variability (HRV): {monitor.features['hrv']:.2f}",
```

```python
horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes, fontsize=14)
```

```python
axs[2].axis('off')
```

*Displaying Skewness and Kurtosis*

```python
axs[3].text(0.5, 0.7, f"Skewness: {monitor.features['skewness']:.2f}",
```

```python
horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes, fontsize=14)
```

```python
axs[3].text(0.5, 0.3, f"Kurtosis: {monitor.features['kurtosis']:.2f}",
```

```python
horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes, fontsize=14)
```

```python
axs[3].axis('off')
```

```python
plt.tight_layout()
```

```python
plt.close(fig)
```

```python
plot_panel = pn.pane.Matplotlib(fig, tight=True)
```

*Empty to allow this folder to be treated as a package and allow communication between files*

# HRD

# __pycache__

# Individual_Tracking

```python
import pandas as pd
```

```python
import numpy as np
```

```python
import json
```

```python
from sklearn.model_selection import train_test_split
```

```python
from datetime import datetime
```

```python
from tensorflow.keras.models import Sequential
```

```python
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Reshape, Masking
```

```python
from tensorflow import keras
```

```python
from keras.callbacks import EarlyStopping
```

```python
class PredictiveTracking:
```

A class for training and handling predictive tracking models for individual users.  Attributes: user_id (str): Unique identifier for the user. model_path (str): Path to the model file. seq_length (int): Length of the input sequence. pred_length (int): Length of the prediction sequence. last_trained_date (datetime): Timestamp of the last model training.

```python
def __init__(self, user_id, preprocessed_data, mode, seq_length=20, pred_length=10):
```

Initializes the PredictiveTracking class.  **Parameter:** user_id: Unique identifier for the user. **Parameter:** preprocessed_data: Preprocessed data for training or testing. **Parameter:** mode: Mode of operation, either 'train' or 'test'. **Parameter:** seq_length: Length of the input sequence, defaults to 20. **Parameter:** pred_length: Length of the prediction sequence, defaults to 10.

```python
self.user_id = user_id
```

```python
self.model_path = f"Models/Individual Tracking & Monitoring/IndiMods/model_{user_id}.h5"
```

```python
self.seq_length = seq_length
```

```python
self.pred_length = pred_length
```

```python
if preprocessed_data is not None:
```

```python
self.load_data(preprocessed_data, mode)
```

```python
def calculate_epochs(self, min_epochs=100, max_epochs=1000):
```

Calculates the number of epochs based on the training samples.  **Parameter:** min_epochs: Minimum number of epochs, defaults to 20. **Parameter:** max_epochs: Maximum number of epochs, defaults to 250. **Returns:** Calculated number of epochs.

*Get the number of training samples*

```python
num_samples = self.X_train.shape[0]
```

*Apply a sigmoid scaling factor*

```python
scaling_factor = 1 / (1 + np.exp(-0.5 * (num_samples - 800)))
```

*Reverse the scaling factor to get an inverse sigmoid*

```python
reverse_scaling_factor = 1 - scaling_factor
```

*Scale the value to the desired range of epochs*

```python
epochs = int(min_epochs + (max_epochs - min_epochs) * reverse_scaling_factor)
```

*Ensure the calculated epochs are within the defined bounds*

```python
epochs = max(min_epochs, min(epochs, max_epochs))
```

```python
return epochs
```

```python
def load_data(self, preprocessed_data, mode='train'):
```

Loads the training and testing data.  **Parameter:** preprocessed_data: Preprocessed data for training or testing. **Parameter:** mode: Mode of operation, either 'train' or 'test', defaults to 'train'.

```python
if mode == 'train':
```

```python
self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(*preprocessed_data, test_size=0.2, random_state=42)
```

```python
print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
```

```python
elif mode == 'test':
```

```python
self.X_test, self.y_test = preprocessed_data
```

```python
print(self.X_test.shape, self.y_test.shape)
```

```python
else:
```

```python
print("Invalid mode. Use 'train' or 'test'.")
```

```python
def load_model(self):
```

Loads a pre-trained model from the file system.  **Returns:** Loaded model and the last trained date, or None if not found.

```python
try:
```

```python
model=keras.models.load_model(self.model_path)
```

```python
with open(f"{str(self.model_path).replace('h5','json')}", "r") as read_file:
```

```python
data = json.load(read_file)
```

```python
self.last_trained_date = datetime.strptime(data['last_trained_date'], "%d-%m-%Y %H:%M:%S.%f")
```

```python
return model, self.last_trained_date
```

```python
except Exception as e:
```

```python
print("No model found --{e}")
```

```python
self.model = None
```

```python
self.last_trained_date = None
```

```python
return
```

```python
def train_model(self):
```

Trains the model using the loaded training data.

```python
try:
```

```python
self.model = Sequential()
```

```python
self.model.add(Masking(mask_value=0., input_shape=(self.seq_length, 27)))
```

*Masking layer*

```python
self.model.add(Bidirectional(LSTM(1024, return_sequences=True), input_shape=(self.seq_length, 17)))
```

*17 features*

```python
self.model.add(Bidirectional(LSTM(1024, return_sequences=True)))
```

```python
self.model.add(Bidirectional(LSTM(1024, return_sequences=True)))
```

```python
self.model.add(Dropout(0.2))
```

```python
self.model.add(Bidirectional(LSTM(1024, return_sequences=True)))
```

```python
self.model.add(Dropout(0.2))
```

```python
self.model.add(Bidirectional(LSTM(512, return_sequences=True)))
```

```python
self.model.add(Dropout(0.2))
```

```python
self.model.add(Bidirectional(LSTM(512, return_sequences=True)))
```

```python
self.model.add(Dropout(0.2))
```

```python
self.model.add(Bidirectional(LSTM(512, return_sequences=False)))
```

```python
self.model.add(Dropout(0.2))
```

```python
self.model.add(Dense(self.pred_length * 2))
```

```python
self.model.add(Reshape((self.pred_length, 2)))
```

*Reshape to (pred_length, 2)*

*Compile with a custom learning rate*

```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)
```

```python
self.model.compile(optimizer=optimizer, loss='mse')
```

*Calculate the number of epochs*

```python
epochs = self.calculate_epochs()
```

*Implement early stopping*

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
```

*Fit the model with validation split*

```python
self.model.fit(self.X_train, self.y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])
```

```python
self.last_trained_date = datetime.now()
```

```python
except Exception as e:
```

```python
print(e)
```

```python
def save_model(self):
```

Saves the trained model to the file system and logs the training date.

```python
self.model.save(self.model_path)
```

```python
print("Model saved")
```

```python
data= self.last_trained_date.strftime("%d/%m/%Y %H:%M:%S")
```

```python
with open(f"{str(self.model_path).replace('h5','json')}", "w") as write_file:
```

```python
json.dump(data, write_file)
```

```python
print("Model logged")
```

```python
import pandas as pd
```

```python
from sklearn.cluster import DBSCAN
```

```python
from sklearn.ensemble import IsolationForest
```

```python
from geopy import distance
```

```python
from geopy.distance import geodesic
```

```python
from geopy.distance import great_circle
```

```python
import numpy as np
```

```python
import glob
```

```python
import os
```

```python
from datetime import datetime,timedelta
```

```python
from tensorflow.keras.models import load_model
```

```python
from PMT_run import PredictiveTracking
```

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

```python
from sklearn.metrics import mean_squared_error,mean_absolute_error
```

```python
import matplotlib.pyplot as plt
```

```python
import pandas as pd
```

```python
import sys
```

```python
import panel as pn
```

```python
sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')
```

```python
broker_address="test.mosquitto.org"
```

```python
topic="Orion_test/Individual Tracking & Monitoring"
```

```python
topic="Orion_test/UTP"
```

```python
from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH
```

```python
Handler=MQDH(broker_address, topic)
```

```python
Reciever=MQDH(broker_address, "Orion_test/UTP")
```

```python
class RealTimeTracking:
```

```python
def __init__(self,user_id):
```

Initializes the RealTimeTracking class with user_id. **Parameter:** user_id: Unique identifier for the user

```python
self.user_id = user_id
```

```python
self.seq_length = 20
```

```python
self.train_date=None
```

```python
self.model_path = f"E:/Dev/Deakin/redbackoperations-T2_2023/Project 1 - Tracking Players and Crowd Monitoring/DataScience/IndividualLSTMs/model_{user_id}.h5"
```

```python
self.predictive_model=None
```

```python
def get_trajectory(self, gps_data):
```

Filters the GPS data to return only the trajectory for the given user. **Parameter:** gps_data: DataFrame containing GPS data **Returns:** DataFrame containing the trajectory for the user

```python
return gps_data[gps_data['user_id'] == self.user_id]
```

```python
def get_direction(self, point1, point2):
```

Calculates the direction from point1 to point2. **Parameter:** point1: Dictionary containing the 'Latitude' and 'Longitude' of the first point **Parameter:** point2: Dictionary containing the 'Latitude' and 'Longitude' of the second point **Returns:** Direction angle in radians

```python
return np.arctan2(point2['Longitude'] - point1['Longitude'], point2['Latitude'] - point1['Latitude'])
```

```python
def get_distance(self, point1, point2):
```

Calculates the distance between two geographical points. **Parameter:** point1: Dictionary containing the 'Latitude' and 'Longitude' of the first point **Parameter:** point2: Dictionary containing the 'Latitude' and 'Longitude' of the second point **Returns:** Distance in meters

```python
return distance.distance((point1['Latitude'],point1['Longitude']),(point2['Latitude'],point2['Longitude'])).meters
```

```python
def get_speed(self, initialpoint,finalpoint,initialtime,finaltime):
```

Calculates the speed between two points given the time taken to travel. **Parameter:** initialpoint: Starting point as a dictionary with 'Latitude' and 'Longitude' **Parameter:** finalpoint: Ending point as a dictionary with 'Latitude' and 'Longitude' **Parameter:** initialtime: Start time as a datetime object **Parameter:** finaltime: End time as a datetime object **Returns:** Speed in meters per second

```python
return self.get_distance(initialpoint,finalpoint) / (finaltime - initialtime).seconds
```

```python
def get_acceleration(self, initialspeed,finalspeed,initialtime,finaltime):
```

Calculates the acceleration given the initial and final speeds and the time taken. **Parameter:** initialspeed: Initial speed in meters per second **Parameter:** finalspeed: Final speed in meters per second **Parameter:** initialtime: Start time as a datetime object **Parameter:** finaltime: End time as a datetime object **Returns:** Acceleration in meters per second squared

```python
return (finalspeed - initialspeed) / (finaltime - initialtime).seconds
```

```python
def get_stops(self, trajectory, time_threshold):
```

```python
stops = []
```

```python
for i in range(1, len(trajectory)):
```

```python
if self.get_distance(trajectory.iloc[i-1], trajectory.iloc[i]) == 0 and \
```

```python
(trajectory.iloc[i]['Datetime'] - trajectory.iloc[i-1]['Datetime']).seconds >= time_threshold:
```

```python
stops.append(trajectory.iloc[i-1])
```

```python
return stops
```

```python
def get_mode(self, speeds, accelerations):
```

```python
avg_speed = np.mean(speeds)
```

```python
avg_acceleration = np.mean(accelerations)
```

```python
if avg_speed < 2:
```

```python
return 'walking'
```

```python
elif avg_speed < 20:
```

```python
return 'cycling'
```

```python
else:
```

```python
return 'driving'
```

```python
def get_frequent_areas(self, trajectory, eps=0.01, min_samples=2):
```

```python
coords = [point[['Latitude', 'Longitude']].tolist() for _, point in trajectory.iterrows()]
```

```python
clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
```

```python
clusters = {}
```

```python
for i, label in enumerate(clustering.labels_):
```

```python
if label != -1:
```

```python
if label not in clusters:
```

```python
clusters[label] = []
```

```python
clusters[label].append(trajectory.iloc[i])
```

```python
return clusters
```

```python
def get_anomalies(self, trajectory):
```

```python
coords = [point[['Latitude', 'Longitude']].tolist() for _, point in trajectory.iterrows()]
```

```python
model = IsolationForest().fit(coords)
```

```python
anomalies = [point for i, point in enumerate(trajectory.iterrows()) if model.predict([coords[i]]) == -1]
```

```python
return anomalies
```

```python
def get_time_feature(self, cluster):
```

```python
return cluster.iloc[0]['Datetime'].hour
```

```python
def create_prediction_data(self, trajectory, common_areas=None):
```

*Process the data similarly to the preprocess_data method*

```python
X, _ = self.preprocess_data(trajectory, common_areas)
```

*Select one sequence*

```python
pred_sequence = X[0]
```

*Replace the first four features with placeholder values (e.g., zeros)*

```python
pred_sequence[:, :4] = 0
```

*Reshape to match the expected input shape*

```python
pred_sequence = pred_sequence.reshape(1, pred_sequence.shape[0], pred_sequence.shape[1])
```

```python
return pred_sequence
```

```python
def get_ground_truth(self, trajectory):
```

*Preprocess the data to get the original features, including the coordinates*

```python
_, y = self.preprocess_data(trajectory)
```

*Select the corresponding ground truth for the first sequence*

```python
ground_truth = y[0]
```

```python
return ground_truth
```

```python
def test_prediction(self, trajectory, common_areas=None):
```

*Create test data for the given trajectory*

```python
test_sequence = self.create_prediction_data(trajectory, common_areas)
```

*Load the trained model*

```python
try:
```

```python
self.model = load_model(self.model_path)
```

```python
except:
```

```python
print("No model found. Please train the model first.")
```

```python
return
```

*Make a prediction using the test sequence*

```python
prediction = self.model.predict(test_sequence)
```

*Retrieve the ground truth (actual future coordinates) for the test sequence*

*Assuming you have a method to obtain the ground truth*

```python
ground_truth = self.get_ground_truth(trajectory)
```

*Compute metrics*

```python
mae = mean_absolute_error(ground_truth, prediction[0])
```

```python
mse = mean_squared_error(ground_truth, prediction[0])
```

*Print the prediction results*

```python
print("Predicted coordinates:")
```

```python
for coords in prediction[0]:
```

```python
print(f"Latitude: {coords[0]}, Longitude: {coords[1]}")
```

*Print the metrics*

```python
print(f"Mean Absolute Error: {mae}")
```

```python
print(f"Mean Squared Error: {mse}")
```

```python
def preprocess_data(self, trajectory, common_areas=None):
```

```python
frequent_areas = list(self.get_frequent_areas(trajectory, 10, 5).values())
```

```python
combined_areas = [area[0] for area in frequent_areas[:10]]
```

*Take the first 10 frequent areas*

*If there are fewer than 10 frequent areas, add common areas*

```python
while len(combined_areas) < 10:
```

```python
if common_areas is not None and len(common_areas) > 0:
```

```python
combined_areas.append({'Latitude': common_areas[0][0], 'Longitude': common_areas[0][1]})
```

```python
common_areas.pop(0)
```

```python
else:
```

```python
combined_areas.append({'Latitude': 0, 'Longitude': 0})
```

*Fill with zeros if not enough areas*

```python
features = []
```

```python
for i, point in enumerate(trajectory.iloc[:-1].to_dict('records')):
```

*Add the coordinates to the feature list along with other features*

```python
feature_vector = [
```

```python
point['Latitude'],
```

```python
point['Longitude'],
```

```python
self.get_speed(point, trajectory.iloc[i+1], point['Datetime'], trajectory.iloc[i+1]['Datetime']),
```

```python
self.get_direction(point, trajectory.iloc[i+1]),
```

*Direction*

```python
point['Datetime'].weekday(),
```

```python
point['Datetime'].hour,
```

```python
point['Datetime'].minute
```

```python
] + [area['Latitude'] for area in combined_areas] + [area['Longitude'] for area in combined_areas]
```

```python
features.append(feature_vector)
```

*Create sequences and future coordinates*

```python
sequences = []
```

```python
future_coordinates = []
```

```python
for i in range(len(features) - self.seq_length - 9):
```

*9 is to accommodate 10 future coordinates*

```python
sequence = features[i:i+self.seq_length]
```

```python
future_coord = [features[j][:2] for j in range(i+self.seq_length, i+self.seq_length+10)]
```

*Take only Latitude and Longitude*

```python
sequences.append(sequence)
```

```python
future_coordinates.append(future_coord)
```

*Convert to NumPy arrays*

```python
X = np.array(sequences)
```

```python
y = np.array(future_coordinates)
```

```python
return X, y
```

```python
def generate_future_coordinates(self, trajectory, common_areas=None, number_of_future_points=100, time_interval_minutes=5):
```

```python
print(trajectory.head(-5))
```

*Get the latest data point from the trajectory*

```python
latest_data_point = trajectory.iloc[-1]
```

*Create a list of future datetimes*

```python
future_datetimes = [latest_data_point['Datetime'] + timedelta(minutes=i * time_interval_minutes) for i in range(1, number_of_future_points + 1)]
```

*Create a DataFrame to hold the future trajectory*

```python
future_trajectory = pd.DataFrame(columns=trajectory.columns)
```

*Populate the future trajectory DataFrame with the latest data point and future datetimes*

```python
for future_datetime in future_datetimes:
```

```python
future_data_point = latest_data_point.copy()
```

```python
future_data_point['Datetime'] = future_datetime
```

```python
future_trajectory = future_trajectory.append(future_data_point, ignore_index=True)
```

```python
print(future_trajectory.head(5))
```

*Create test data for the given trajectory and future datetimes*

```python
pred_sequence = self.create_prediction_data(future_trajectory, common_areas)
```

*Load the trained model*

```python
try:
```

```python
self.model = load_model(self.model_path)
```

```python
except:
```

```python
print("No model found. Please train the model first.")
```

```python
return
```

*Make a prediction using the test sequence*

```python
prediction = self.model.predict(pred_sequence)
```

*Convert the prediction to a list of coordinates (latitude, longitude)*

```python
predicted_coordinates = [(coords[0], coords[1]) for coords in prediction[0]]
```

```python
return predicted_coordinates
```

```python
def train_personalised_model(self, trajectory_data, retrain=False):
```

```python
preprocessed_data = self.preprocess_data(trajectory_data)
```

```python
try:
```

```python
model=load_model(self.model_path)
```

```python
except:
```

```python
model=None
```

```python
if retrain and model is not None:
```

```python
print("Retraining model......")
```

```python
self.predictive_model = PredictiveTracking(self.user_id, preprocessed_data,'train')
```

```python
self.predictive_model.train_model()
```

```python
self.predictive_model.save_model()
```

```python
return
```

```python
elif not retrain and model is not None:
```

```python
print("Model already trained. Use retrain=True to retrain the model.")
```

```python
return
```

```python
elif model is None:
```

```python
print("No model found. Training new model...")
```

```python
self.predictive_model = PredictiveTracking(self.user_id, preprocessed_data,'train')
```

```python
self.predictive_model.train_model()
```

```python
self.predictive_model.save_model()
```

```python
return
```

*this should only be used during testing --some modifications needed to use it in production*

```python
def read_plt(file_path, user_id):
```

Reads a plt file and returns it as a DataFrame, adding a user_id column. **Parameter:** file_path: Path to the plt file **Parameter:** user_id: Unique identifier for the user **Returns:** DataFrame containing the plt file data with added user_id

```python
columns = ['Latitude', 'Longitude', 'Reserved', 'Altitude', 'NumDays', 'Date', 'Time']
```

```python
data = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
```

```python
data['Altitude'] = data['Altitude'] * 0.3048
```

```python
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
```

```python
data.drop(columns=['Date', 'Time'], inplace=True)
```

```python
data['user_id'] = user_id
```

*Add user_id to the DataFrame*

```python
return data
```

*common_areas = [(lat1, lon1), (lat2, lon2), ...]*

*Example common areas i.e. landmarks,coffee shops*

*Empty to allow this folder to be treated as a package and allow communication between files*

# __pycache__

# Overcrowding_Detection

```python
import numpy as np
```

```python
import pandas as pd
```

```python
import matplotlib.pyplot as plt
```

```python
import matplotlib.cm as cm
```

```python
from sklearn.cluster import DBSCAN
```

```python
import seaborn as sns
```

```python
import folium
```

```python
from folium.plugins import HeatMap
```

```python
import panel as pn
```

```python
import sys
```

```python
sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')
```

```python
broker_address="test.mosquitto.org"
```

```python
topic="Orion_test/Overcrowding Detection"
```

```python
from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH
```

```python
Handler=MQDH(broker_address, topic)
```

```python
def process_data(gps_data):
```

Processes the GPS data using DBSCAN clustering and plots the clusters.  **Parameter:** gps_data: A numpy array with GPS coordinates in the format [[latitude1, longitude1], [latitude2, longitude2], ...].

*Using DBSCAN to cluster the data*

```python
dbscan = DBSCAN(eps=0.02, min_samples=15, metric='euclidean')
```

```python
clusters = dbscan.fit_predict(gps_data)
```

```python
df = pd.DataFrame({'Latitude': gps_data[:, 0], 'Longitude': gps_data[:, 1], 'Cluster': clusters})
```

```python
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
```

```python
colors = cm.rainbow(np.linspace(0, 1, num_clusters))
```

```python
figure=plt.figure(figsize=(10, 10))
```

```python
for cluster_id, color in zip(set(clusters), colors):
```

```python
if cluster_id == -1:
```

```python
continue
```

```python
cluster_points = df[df['Cluster'] == cluster_id]
```

```python
plt.scatter(cluster_points['Longitude'], cluster_points['Latitude'], c=[color], label=f'Cluster {cluster_id}')
```

```python
plt.title('Clusters of Points')
```

```python
plt.xlabel('Longitude')
```

```python
plt.ylabel('Latitude')
```

```python
plt.legend()
```

```python
plt.show()
```

```python
return  df,figure
```

*Initialize empty arrays for latitudes and longitudes*

```python
latitudes = []
```

```python
longitudes = []
```

*reading latitudes and longitudes from the VirtualCrowd_Test_Cleaned.csv*

*change this line depending on the file path*

```python
df = pd.read_csv(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Clean Datasets\VD4.csv')
```

```python
selected_time_data = df[df["Time"] == '22:22:01']
```

```python
time = df[df["Time"].isin(['23:39:33'])].reset_index(drop=True)
```

```python
latitudes_list = time[" Longitude Degrees"].tolist()
```

```python
longitudes_list = time[" Latitude Degrees"].tolist()
```

*remove "Time" to facilitate the processing*

```python
df.drop("Time", axis=1)
```

```python
latitudes = latitudes_list
```

```python
longitudes = longitudes_list
```

*Processing the data and plotting*

```python
gps_data = np.array([latitudes, longitudes]).T
```

```python
cluster_data,fig1=process_data(gps_data)
```

*Creating a DataFrame with the latitude and longitude data*

```python
heatmap_data = pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes})
```

*Plotting the heatmap using Seaborn's kdeplot function*

```python
figure2=plt.figure(figsize=(10, 10))
```

```python
sns.kdeplot(x='Longitude', y='Latitude', data=heatmap_data, cmap='Reds', fill=True)
```

```python
plt.title('Heatmap of Points')
```

```python
plt.xlabel('Longitude')
```

```python
plt.ylabel('Latitude')
```

```python
plt.show()
```

*Calculate the mean latitude and longitude for centering the map*

```python
mean_latitude = selected_time_data[" Latitude Degrees"].mean()
```

```python
mean_longitude = selected_time_data[" Longitude Degrees"].mean()
```

*Create a base map, centered at the mean coordinates*

```python
base_map = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=13)
```

*Create a list of [lat, lon] pairs for the heatmap*

```python
heatmap_data = [[lat, lon] for lat, lon in zip(selected_time_data[" Latitude Degrees"], selected_time_data[" Longitude Degrees"])]
```

*Add the heatmap layer to the base map with adjusted parameters*

```python
HeatMap(heatmap_data, radius=15, max_zoom=13, blur=15).add_to(base_map)
```

*Save the map to an HTML file (optional)*

*change this line depending on the file path*

```python
base_map.save(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Overcrowding_Detection\heatmap.html')
```

*Handler.send_data(cluster_data)#send the data to the broker*

*plt.close(fig1)*

*plt.close(figure2)*

```python
plot_panel = pn.pane.Matplotlib(fig1, tight=True)
```

```python
plot_panel2 = pn.pane.Matplotlib(figure2, tight=True)
```

*Empty to allow this folder to be treated as a package and allow communication between files*

# __pycache__

# variables

