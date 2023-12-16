import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Reshape, Masking
from tensorflow import keras   
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import random
import math


class ReshuffleCallback(keras.callbacks.Callback):
    def __init__(self, X, y, split_ratio=0.8):
        super(ReshuffleCallback, self).__init__()
        self.X = X
        self.y = y
        self.split_ratio = split_ratio
        
    def on_epoch_begin(self, epoch, logs=None):
        # Shuffle the entire dataset
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        X_shuffled = self.X[indices]
        y_shuffled = self.y[indices]
        
        # Split the shuffled data using the manual method
        split_idx = int(self.split_ratio * X_shuffled.shape[0])
        self.model.X_train = X_shuffled[:split_idx]
        self.model.y_train = y_shuffled[:split_idx]
        self.model.X_test = X_shuffled[split_idx:]
        self.model.y_test = y_shuffled[split_idx:]
        
# Create an instance of the callback with the data and desired split ratio

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
    def __init__(self, user_id, preprocessed_data, mode, seq_length=60, pred_length=10):
        """
        Initializes the PredictiveTracking class.

        :param user_id: Unique identifier for the user.
        :param preprocessed_data: Preprocessed data for training or testing.
        :param mode: Mode of operation, either 'train' or 'test'.
        :param seq_length: Length of the input sequence, defaults to 20.
        :param pred_length: Length of the prediction sequence, defaults to 10.
        """
        self.user_id = user_id
        
        self.model_path = f"/home/sam/Desktop/DataScience/Models/Individual_Tracking/IndividualLSTMs/model_{user_id}.h5"
        self.last_trained_date=datetime.now()     
        self.seq_length = seq_length
        self.pred_length = pred_length
        if preprocessed_data is not None:
         self.load_data(preprocessed_data,0.8, mode)
         
##using simulated anneaing to find the best hyperparameters
        self.initial_params = {
        'seq_length': 60,
        'lstm_units': [12,24,48,96,192],
        'dropout_rates': [0.8, 0.6, 0.4 , 0.2 , 0.2],
        'learning_rate': 0.0001
        }
        self.initial_temperature = 10.0
        self.cooling_rate = 0.995  # Typical values are between 0.8 and 0.999
        self.best_params, self.best_cost = None, None
        self.best_model = None

    def get_neighbor(self,params):
        new_params = params.copy()
        
        # Randomly adjust learning rate
        factor = 1 + (random.random() - 0.5) / 10
        new_params['learning_rate'] *= factor
        
        # Randomly adjust LSTM layers and units
        if random.random() > 0.5 and len(new_params['lstm_units']) < 5:  # Let's limit to a max of 5 LSTM layers for this example
            # Add a new LSTM layer with a random number of units (between 512 and 2048 for this example)
            new_params['lstm_units'].append(random.randint(12,192))
            new_params['dropout_rates'].append(0.1 + random.random() * 0.4)  # Random dropout rate between 0.1 and 0.5
        elif len(new_params['lstm_units']) > 1:  # Ensure at least one LSTM layer remains
            # Remove the last LSTM layer
            new_params['lstm_units'].pop()
            new_params['dropout_rates'].pop()
        
        # Randomly adjust the units of existing LSTM layers
        for i in range(len(new_params['lstm_units'])):
            new_params['lstm_units'][i] = random.randint(128, 2048)  # Randomly adjust number of units
        
        return new_params


    def simulated_annealing(self,initial_params, initial_temperature, cooling_rate, num_iterations=10):
        current_params = initial_params
        current_cost = self.objective_function(current_params)
        
        best_params = current_params
        best_cost = current_cost
        
        temperature = initial_temperature
        
        for i in range(num_iterations):
            # Generate a neighboring solution
            neighbor_params = self.get_neighbor(current_params)
            neighbor_cost = self.objective_function(neighbor_params)
            
            # If the new solution is better or accepted with a certain probability, update the current solution
            cost_diff = neighbor_cost - current_cost
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_params, current_cost = neighbor_params, neighbor_cost
                
                # Update the best solution found so far
                if current_cost < best_cost:
                    best_cost, best_params = current_cost, current_params
            
            # Reduce the temperature
            temperature *= cooling_rate
        
        return best_params, best_cost


    def objective_function(self,params):
       
        val_loss, val_accuracy = self.train_with_parameters(lstm_units=params['lstm_units'], dropout_rates=params['dropout_rates'], learning_rate=params['learning_rate'])
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        # Compute the composite objective
        composite_objective = 0.9 * val_loss + 0.1 * (1 - val_accuracy)
        return composite_objective

    
    def calculate_epochs(self, min_epochs=50, max_epochs=500):
        """
        Calculates the number of epochs based on the training samples.

        :param min_epochs: Minimum number of epochs, defaults to 50.
        :param max_epochs: Maximum number of epochs, defaults to 300.
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

            
    def load_data(self, preprocessed_data, split_ratio=0.8, mode='train'):
        """
        Loads the training and testing data using a time-based split.

        :param preprocessed_data: Preprocessed data for training or testing.
        :param split_ratio: Ratio of data to be used for training. For example, 0.8 means 80% for training and 20% for testing.
        :param mode: Mode of operation, either 'train' or 'test', defaults to 'train'.
        """
        
        self.X, self.y = preprocessed_data
        
        # Calculate the split index based on the split_ratio
        split_idx = int(len(self.X) * split_ratio)
        
        if mode == 'train':
            self.X_train, self.X_test = self.X[:split_idx], self.X[split_idx:]
            self.y_train, self.y_test = self.y[:split_idx], self.y[split_idx:]
            
            print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        elif mode == 'test':
            self.X_test, self.y_test = self.X, self.y
            print(self.X_test.shape, self.y_test.shape)
        else: 
            print("Invalid mode. Use 'train' or 'test'.")

    # This is just a representation of the modified function. Actual integration would need adjustment based on the entire class structure.

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


    def train_with_parameters(self, lstm_units=[12,24,48,96,192], dropout_rates=[0.5, 0.2, 0.2, 0.0 , 0.0], learning_rate=0.001):
        """Trains the model using the loaded training data."""
        try:
            self.model = Sequential()
            self.model.add(Masking(mask_value=0., input_shape=(self.seq_length, 30))) # Masking layer
            print("building inner layers")
            # Dynamically adding LSTM layers based on lstm_units list
            for i, units in enumerate(lstm_units):
                self.model.add(Bidirectional(LSTM(units, return_sequences=True if i < len(lstm_units) - 1 else False)))
                self.model.add(Dropout(dropout_rates[i]))
            
            print("building output layer")
            self.model.add(Dense(self.pred_length * 2))
            self.model.add(Reshape((self.pred_length, 2))) # Reshape to (pred_length, 2)
            reshuffle_callback = ReshuffleCallback(self.X,self.y, split_ratio=0.8)
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
            
            epochs = self.calculate_epochs()
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

            print("training model")
            history = self.model.fit(self.X_train, self.y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stopping,reduce_lr,reshuffle_callback])
            print("history",history.history)
            self.last_trained_date = datetime.now()
            
            return history.history['val_loss'][-1], history.history['val_accuracy'][-1]
        except Exception as e:
            print(e)
            print("Error training model")
            return None, None


    def train_model(self):
        self.best_params, self.best_cost = self.simulated_annealing(self.initial_params, self.initial_temperature, self.cooling_rate)
        print("Best Hyperparameters:", self.best_params)
        print("Best Cost:", self.best_cost)
        
        # Train the model with the best hyperparameters
        self.train_with_parameters(lstm_units=self.best_params['lstm_units'], 
                                dropout_rates=self.best_params['dropout_rates'], 
                                learning_rate=self.best_params['learning_rate'])
        self.save_model()
        return self.model

    def save_model(self):
        """
        Saves the trained model to the file system and logs the training date and best hyperparameters.
        """
        self.model.save(self.model_path)
        print("Model saved")

        log_data = {
            "last_trained_date": self.last_trained_date.strftime("%d/%m/%Y %H:%M:%S"),
            "best_lstm_units": self.best_params['lstm_units'],
            "best_dropout_rates": self.best_params['dropout_rates'],
            "best_learning_rate": self.best_params['learning_rate']
        }

        with open(f"{str(self.model_path).replace('h5','json')}", "w") as write_file:
            json.dump(log_data, write_file)
            print("Model logged")
