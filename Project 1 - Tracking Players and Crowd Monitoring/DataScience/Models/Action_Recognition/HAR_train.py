import os
import glob
from sklearn.preprocessing import MinMaxScaler
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.layers import LSTM,Bidirectional
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Masking,Dense,Dropout
from tensorflow.keras.optimizers import Adam
import keras
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)



wandb.init(
    # set the wandb project where this run will be logged
    project="Project Orion",

    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 24,
        "activation_1": "tanh",
        "layer_2": 24,
        "activation_2": "tanh",
        "layer_3": 12,
        "activation_3": "tanh",
        "layer_3": 12,
        "activation_3": "tanh",
        "optimizer": "adam",
        "layer_4": 6,
        "activation_4": "softmax",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 30,
        "batch_size": 16
    }
)

class TrainingClass:
    def __init__(self, data_path=r'E:\Test\A_DeviceMotion_data'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
        print("Num TPUs Available: ", len(tf.config.list_physical_devices('TPU')))

    def load_data(self):
        sequences = []
        labels = []
        max_length = 0

        # Determine max_length without loading all data
        for folder_path in glob.glob(os.path.join(self.data_path, '*_*')):
            for subject_file in glob.glob(os.path.join(folder_path, 'sub_*.csv')):
                with open(subject_file, 'r') as file:
                    row_count = sum(1 for row in file)-1
                    max_length = max(max_length, row_count)

        # Loop through all files in the data path and load with padding
        for folder_path in glob.glob(os.path.join(self.data_path, '*_*')):
            action = os.path.basename(folder_path).split('_')[0]
            for subject_file in glob.glob(os.path.join(folder_path, 'sub_*.csv')):
                df = pd.read_csv(subject_file)
                df = df.iloc[:, 1:]
                df.fillna(0, inplace=True)
                    # Compute statistical features
                df['mean'] = df.mean(axis=1)
                df['std'] = df.std(axis=1)
                df['min'] = df.min(axis=1)
                df['max'] = df.max(axis=1)
                df['UAx_lag_1'] = df['userAcceleration.x'].shift(1)
                df['UAx_diff_1'] = df['userAcceleration.x'].diff(1)              
                df['UAy_lag_1'] = df['userAcceleration.y'].shift(1)
                df['UAy_diff_1'] = df['userAcceleration.y'].diff(1)               
                df['UAz_lag_1'] = df['userAcceleration.z'].shift(1)
                df['UAz_diff_1'] = df['userAcceleration.z'].diff(1)
                df['UA_magnitude'] = np.sqrt(np.abs(df['userAcceleration.x']**2 + df['userAcceleration.y']**2 + df['userAcceleration.z']**2))
                
              
                df['Rx_lag_1'] = df['rotationRate.x'].shift(1)
                df['Rx_diff_1'] = df['rotationRate.x'].diff(1)                
                df['Ry_lag_1'] = df['rotationRate.y'].shift(1)
                df['Ry_diff_1'] = df['rotationRate.y'].diff(1)               
                df['Rz_lag_1'] = df['rotationRate.z'].shift(1)
                df['Rz_diff_1'] = df['rotationRate.z'].diff(1)
                df['R_magnitude'] = np.sqrt(np.abs(df['rotationRate.x']**2 + df['rotationRate.y']**2 + df['rotationRate.z']**2))
                
               
                df['Gx_lag_1'] = df['gravity.x'].shift(1)
                df['Gx_diff_1'] = df['gravity.x'].diff(1)
                df['Gy_lag_1'] = df['gravity.y'].shift(1)
                df['Gy_diff_1'] = df['gravity.y'].diff(1)
                df['Gz_lag_1'] = df['gravity.z'].shift(1)
                df['Gz_diff_1'] = df['gravity.z'].diff(1)
                df['G_magnitude'] = np.sqrt(np.abs(df['gravity.x']**2 + df['gravity.y']**2 + df['gravity.z']**2))
                
                
                df['Ax_lag_1'] = df['attitude.roll'].shift(1)
                df['Ax_diff_1'] = df['attitude.roll'].diff(1)                
                df['Ay_lag_1'] = df['attitude.pitch'].shift(1)
                df['Ay_diff_1'] = df['attitude.pitch'].diff(1)                
                df['Az_lag_1'] = df['attitude.yaw'].shift(1)
                df['Az_diff_1'] = df['attitude.yaw'].diff(1)
                df['A_magnitude'] = np.sqrt(np.abs(df['attitude.roll']**2 + df['attitude.pitch']**2 + df['attitude.yaw']**2))
               
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.fillna(0, inplace=True)
               
                # Pad the DataFrame with zeros up to max_length rows
                padded_df = pd.DataFrame(index=range(max_length), columns=df.columns).fillna(0)
                padded_df.iloc[:len(df)] = df.values
                sequences.append(padded_df.values)
                labels.append(action)


        X = np.stack(sequences)
        y = LabelEncoder().fit_transform(labels)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = self.X_train.astype('float')
        self.X_test = self.X_test.astype('float')
        print(self.X_train.shape, self.X_train.dtype)
        print(self.y_train.shape, self.y_train.dtype)
        print(self.X_test.shape, self.X_test.dtype)
        print(self.y_test.shape, self.y_test.dtype)
        print("Unique labels in y_train:", np.unique(self.y_train))
        print("Unique labels in y_test:", np.unique(self.y_test))

        
    def preprocess_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = np.array([self.scaler.fit_transform(x) for x in self.X_train])
        self.X_test = np.array([self.scaler.transform(x) for x in self.X_test])


    def train_model(self):
        self.model = Sequential()
        self.model.add(Masking(mask_value=0., input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Bidirectional(LSTM(256, return_sequences=True)))  
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(256, return_sequences=True)))  
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(128, return_sequences=False)))  
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dense(6, activation='softmax'))
        class_weights = class_weight.compute_class_weight('balanced',classes=
                                                 np.unique(self.y_train),y=
                                                 self.y_train)
        class_weights = dict(enumerate(class_weights))

        
       
       
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=15)
        lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    
        

        self.model.fit(self.X_train, self.y_train, epochs=30, batch_size=16, validation_data=(self.X_test, self.y_test), callbacks=[early_stopping,lr_decay,WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")],class_weight=class_weights)

    def save_model(self, file_path='HARIoT.h5'):
        self.model.save(file_path)

TC = TrainingClass()
TC.load_data()
TC.preprocess_data()
TC.train_model()
TC.save_model()
