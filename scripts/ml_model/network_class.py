import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout ,Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, History
from scripts.config import PROJECT_ROOT

class NeuralNetworkManager:
    def __init__(self, model_name, folder_name):
        """
        Initialize the Neural Network Manager.
        
        Args:
            data (Dataset): Dataset object containing training and testing data
            model_name (str): Name of the model
            folder (str, optional): Folder to save models. Defaults to "smaller".
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        """
        self.folder = os.path.join(PROJECT_ROOT,folder_name)
        self.model = None
        self.trained = False
        self.model_name = model_name
        self.compiled = False
        self.history = None
        self.save_to = os.path.join(self.folder, self.model_name)
        self.ensure_folder_exists(self.save_to)

    def ensure_folder_exists(self, path):
        """
        Ensure the specified directory exists.
        
        Args:
            path (str): Directory path to create
        """
        os.makedirs(path, exist_ok=True)

    def create_mlp_network(self, x_train_shape:tuple, dense_list:list, model_name=None):
        """
        Create neural network model.
        
        Args:
            dense_list (list): List of neurons for hidden layers
            model_name (str, optional): Name of the model. Defaults to None.
        
        Raises:
            ValueError: If input data is invalid or dense_list is not provided
        """
        if len(x_train_shape) >= 3:
            raise ValueError("Invalid input data shape for model creation. Must be 2d vector")
        
        if not dense_list:
            raise ValueError("Please specify a list of neurons for hidden layers. "
                             "E.g., [10, 10] for two layers with 10 neurons each.")
        
        if model_name: self.model_name = model_name
        input_layer= Input(shape=x_train_shape[1:])
        x= input_layer
        for units in dense_list:
            x = Dense(units)(x)
            x = Dropout(0.2)(x)
        
        x = LeakyReLU(alpha=0.3)(x)
        output_layer = Dense(1)(x)
        self.model = Model(input_layer, output_layer, name=model_name)
        print(f"Model '{self.model_name}' created.")
    
    def cnn_network(self,x_train_shape:tuple, filter_list:list,model_name=None, kernel_size=3,activation='relu'):
        """
        Create neural network model.
        
        Args:
            dense_list (list): List of neurons for hidden layers
            model_name (str, optional): Name of the model. Defaults to None.
        
        Raises:
            ValueError: If input data is invalid or dense_list is not provided
        """
        if not filter_list:
            raise ValueError("Please specify a list of neurons for hidden layers. "
                             "E.g., [10, 10] for two layers with 10 neurons each.")
        input_layer= Input(shape=x_train_shape[1:])
        layer=input_layer
        for filter in filter_list:
            layer = Conv1D(filter, kernel_size=kernel_size, activation=activation, padding='same')(layer)
            layer = MaxPooling1D(pool_size=2, padding='same')(layer)
        layer=Flatten()(layer)
        layer=Dense(8,activation='relu')(layer)
        layer=Dense(4,activation='relu')(layer)
        output_layer = Dense(1)(layer)
        self.model = Model(input_layer, output_layer, name=model_name)
        print(f"Model '{self.model_name}' created.")       

             

    def compile_model(self, summarize=False, save=False, optimizer=None, learning_rate=1e-4):
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
        if optimizer != None: 
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=self.optimizer, loss="mse", metrics=['mse'])
        self.compiled = True
        
        if summarize:
            self.model.summary()

        if save:
            self.model.save(os.path.join(self.save_to, f"{self.model_name}.keras"))

    def fit_model(self, x_train, y_train, epochs=250, callbacks=[],save_checkpoint=False, save_model=True, batch_size=32):
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

        
        callbacks = callbacks
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
            validation_split=0.3, 
            callbacks=callbacks
        )
        self.trained = True

        if save_model:
            model_file=os.path.join(self.save_to,f"{self.model_name}_{epochs}.keras")
            self.ensure_folder_exists(self.save_to)
            
            self.model.save(model_file)
            
            # Save training history
            history_path = os.path.join(self.save_to,
                f"{self.model_name}_{epochs}_history.json"
            )
            json.dump(self.history.history, open(history_path, 'w'))

    def predict(self,x_test,y_test):
        """
        Make predictions on test data.
        
        Returns:
            tuple: Predictions and Mean Squared Error
        
        Raises:
            AttributeError: If model is not ready for prediction
        """
        if not self.trained or not self.compiled:
            raise AttributeError("Model must be trained and compiled before prediction.")

        predictions = self.model.predict(x_test).reshape(-1)
        mse = ((y_test - predictions) ** 2).mean()
        
        print(f"Mean Squared Error: {mse}")
        return predictions, mse

    def load_model(self, epochs):
        """
        Load a previously saved model.
        
        Args:
            model_name (str): Name of the model to load
            epochs (int): Number of epochs in the saved model
        
        Raises:
            FileNotFoundError: If specified model does not exist on path 
        """
        path = os.path.join(self.folder, self.model_name, f"{self.model_name}_{epochs}.keras")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        self.model = load_model(path)
        self.compiled = True
        self.trained = True
