import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, History

def half_vectorize(array: np.ndarray):
    """
    Flatten upper triangular part of a symmetric matrix array.
    
    Args:
        array (np.ndarray): Input array to be half vectorized
    
    Returns:
        np.ndarray: Half vectorized array
    """
    if len(array.shape) < 3:
        return array
    
    half_vectors = []
    array = array.reshape(array.shape[0], array.shape[1],array.shape[2])
    for X in array:
        v = X[np.triu_indices(X.shape[0], k=0)]
        half_vectors.append(v)
    return np.array(half_vectors)

class NeuralNetworkManager:
    def __init__(self, data, model_name, folder="smaller", learning_rate=1e-4):
        """
        Initialize the Neural Network Manager.
        
        Args:
            data (Dataset): Dataset object containing training and testing data
            model_name (str): Name of the model
            folder (str, optional): Folder to save models. Defaults to "smaller".
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        """
        self.data = data
        self.folder = folder
        self.learning_rate = learning_rate
        self.model = None
        self.trained = False
        self.model_name = model_name
        self.optimizer = Adam(learning_rate=learning_rate)
        self.compiled = False
        self.history = None

    def ensure_folder_exists(self, path):
        """
        Ensure the specified directory exists.
        
        Args:
            path (str): Directory path to create
        """
        os.makedirs(path, exist_ok=True)

    def create_network(self, dense_list, model_name=None):
        """
        Create neural network model.
        
        Args:
            dense_list (list): List of neurons for hidden layers
            model_name (str, optional): Name of the model. Defaults to None.
        
        Raises:
            ValueError: If input data is invalid or dense_list is not provided
        """
        if self.data.x_train is None or len(self.data.x_train.shape) < 3:
            raise ValueError("Invalid input data shape for model creation.")
        
        if not dense_list:
            raise ValueError("Please specify a list of neurons for hidden layers. "
                             "E.g., [10, 10] for two layers with 10 neurons each.")
        
        if model_name: self.model_name = model_name
        input_layer = Input(shape=self.data.x_train.shape[1:])
        x = Flatten()(input_layer)
        for units in dense_list:
            x = Dense(units)(x)
            x = LeakyReLU(alpha=0.3)(x)
        
        output_layer = Dense(1)(x)
        self.model = Model(input_layer, output_layer, name=model_name)
        print(f"Model '{self.model_name}' created.")

    def compile_model(self, summarize=False, save=False, optimizer=None):
        """
        Compile the neural network model.
        
        Args:
            summarize (bool, optional): Print model summary. Defaults to False.
            save (bool, optional): Save compiled model. Defaults to False.
        
        Raises:
            ValueError: If no model has been defined
        """
        if self.model is None:
            raise ValueError("No model has been defined. Use `create_network` first.")
        if optimizer != None: self.optimizer = optimizer
        self.model.compile(optimizer=self.optimizer, loss="mse", metrics=['mse'])
        self.compiled = True
        
        if summarize:
            self.model.summary()

        if save:
            dir_name = os.path.join(self.folder, self.model_name)
            self.ensure_folder_exists(dir_name)
            self.model.save(os.path.join(dir_name, f"{self.model_name}.h5"))

    def fit_model(self, epochs=250, save_checkpoint=False, save_model=True, batch_size=32):
        """
        Train the neural network model.
        
        Args:
            epochs (int, optional): Number of training epochs. Defaults to 250.
            save_checkpoint (bool, optional): Save model checkpoints. Defaults to False.
            save_model (bool, optional): Save final model. Defaults to True.
            batch_size (int, optional): Training batch size. Defaults to 32.
        
        Raises:
            ValueError: If model is not compiled or created
        """
        if not self.compiled or self.model is None:
            raise ValueError("Model must be created and compiled before training.")

        x_train, y_train = self.data.x_train, self.data.y_labeled_train[:, 1]
        
        callbacks = []
        if save_checkpoint:
            checkpoint_path = os.path.join(self.folder, self.model_name, 
                                           "cp-{epoch:04d}.ckpt")
            self.ensure_folder_exists(os.path.dirname(checkpoint_path))
            callbacks.append(ModelCheckpoint(
                filepath=checkpoint_path, 
                save_weights_only=True, 
                save_freq="epoch", 
                verbose=1
            ))

        self.history = self.model.fit(
            x_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            shuffle=True, 
            validation_split=0.2, 
            callbacks=callbacks
        )
        self.trained = True

        if save_model:
            save_path = os.path.join(
                self.folder, 
                self.model_name, 
                f"{self.model_name}_{epochs}.h5"
            )
            self.model.save(save_path)
            
            # Save training history
            history_path = os.path.join(
                self.folder, 
                self.model_name, 
                f"{self.model_name}_{epochs}_history.json"
            )
            json.dump(self.history.history, open(history_path, 'w'))

    def predict(self):
        """
        Make predictions on test data.
        
        Returns:
            tuple: Predictions and Mean Squared Error
        
        Raises:
            AttributeError: If model is not ready for prediction
        """
        if not self.trained or not self.compiled:
            raise AttributeError("Model must be trained and compiled before prediction.")

        x_test, y_test = self.data.x_test, self.data.y_labeled_test[:, 1]
        x_test = flatten_array(x_test)
        
        predictions = self.model.predict(x_test).reshape(-1)
        mse = ((y_test - predictions) ** 2).mean()
        
        print(f"Mean Squared Error: {mse}")
        return predictions, mse

    def load_model(self, model_name, epochs):
        """
        Load a previously saved model.
        
        Args:
            model_name (str): Name of the model to load
            epochs (int): Number of epochs in the saved model
        
        Raises:
            FileNotFoundError: If specified model does not exist on path 
        """
        path = os.path.join(self.folder, model_name, f"{model_name}_{epochs}.h5")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        self.model = load_model(path)
        self.model_name = model_name
        self.compiled = True
        self.trained = True