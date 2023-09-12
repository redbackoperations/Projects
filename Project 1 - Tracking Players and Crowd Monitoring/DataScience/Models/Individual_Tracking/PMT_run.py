import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Reshape, Masking
from tensorflow import keras   
from keras.callbacks import EarlyStopping 

class PredictiveTracking:
    """
    A class for training and handling predictive tracking models for individual users.
    
    Attributes:
        user_id (str): Unique identifier for the user.
        model_path (str): Path to the model file.
        seq_length (int): Length of the input sequence.
        pred_length (int): Length of the prediction sequence.
        last_trained_date (datetime): Timestamp of the last model training.
    """
    def __init__(self, user_id, preprocessed_data, mode, seq_length=20, pred_length=10):
        """
        Initializes the PredictiveTracking class.

        :param user_id: Unique identifier for the user.
        :param preprocessed_data: Preprocessed data for training or testing.
        :param mode: Mode of operation, either 'train' or 'test'.
        :param seq_length: Length of the input sequence, defaults to 20.
        :param pred_length: Length of the prediction sequence, defaults to 10.
        """
        self.user_id = user_id
        self.model_path = f"Models/Individual Tracking & Monitoring/IndiMods/model_{user_id}.h5"
                       
        self.seq_length = seq_length
        self.pred_length = pred_length
        if preprocessed_data is not None:
         self.load_data(preprocessed_data, mode)
         




    def calculate_epochs(self, min_epochs=20, max_epochs=250):
        """
        Calculates the number of epochs based on the training samples.

        :param min_epochs: Minimum number of epochs, defaults to 20.
        :param max_epochs: Maximum number of epochs, defaults to 250.
        :return: Calculated number of epochs.
        """
        # Get the number of training samples
        num_samples = self.X_train.shape[0]

        # Apply a sigmoid scaling factor
        scaling_factor = 1 / (1 + np.exp(-0.5 * (num_samples - 800)))

        # Reverse the scaling factor to get an inverse sigmoid
        reverse_scaling_factor = 1 - scaling_factor

        # Scale the value to the desired range of epochs
        epochs = int(min_epochs + (max_epochs - min_epochs) * reverse_scaling_factor)

        # Ensure the calculated epochs are within the defined bounds
        epochs = max(min_epochs, min(epochs, max_epochs))

        return epochs

        
    def load_data(self, preprocessed_data, mode='train'):
        """
        Loads the training and testing data.

        :param preprocessed_data: Preprocessed data for training or testing.
        :param mode: Mode of operation, either 'train' or 'test', defaults to 'train'.
        """
        if mode == 'train':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(*preprocessed_data, test_size=0.2, random_state=42)
            print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        elif mode == 'test':
            self.X_test, self.y_test = preprocessed_data
            print(self.X_test.shape, self.y_test.shape)
        else: 
            print("Invalid mode. Use 'train' or 'test'.")

    def load_model(self):
        """
        Loads a pre-trained model from the file system.

        :return: Loaded model and the last trained date, or None if not found.
        """
        try:
            model=keras.models.load_model(self.model_path)
            with open(f"{str(self.model_path).replace('h5','json')}", "r") as read_file:
                data = json.load(read_file)
            self.last_trained_date = datetime.strptime(data['last_trained_date'], "%d-%m-%Y %H:%M:%S.%f")
            return model, self.last_trained_date
        except Exception as e:
            print("No model found --{e}")
            self.model = None
            self.last_trained_date = None
            return

    def train_model(self):
        """
        Trains the model using the loaded training data.
        """
        try:
            self.model = Sequential()
            self.model.add(Masking(mask_value=0., input_shape=(self.seq_length, 27))) # Masking layer
            self.model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(self.seq_length, 17))) # 17 features
            self.model.add(Dropout(0.2))
            self.model.add(Bidirectional(LSTM(256, return_sequences=True)))
            self.model.add(Dropout(0.2))
            self.model.add(Bidirectional(LSTM(256, return_sequences=True)))
            self.model.add(Dropout(0.2))
            self.model.add(Bidirectional(LSTM(256, return_sequences=False)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.pred_length * 2))
            self.model.add(Reshape((self.pred_length, 2))) # Reshape to (pred_length, 2)
             # Compile with a custom learning rate
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            self.model.compile(optimizer=optimizer, loss='mse')
            # Calculate the number of epochs
            epochs = self.calculate_epochs()
            # Implement early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            # Fit the model with validation split
            self.model.fit(self.X_train, self.y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

            self.last_trained_date = datetime.now()
        except Exception as e:
            print(e)

    def save_model(self):
        """
        Saves the trained model to the file system and logs the training date.
        """
        self.model.save(self.model_path)
        print("Model saved")
        data= self.last_trained_date.strftime("%d/%m/%Y %H:%M:%S")
        with open(f"{str(self.model_path).replace('h5','json')}", "w") as write_file:
                json.dump(data, write_file)
                print("Model logged")
        
