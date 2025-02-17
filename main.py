from scripts.ml_model.network_class import NeuralNetworkManager
from scripts.data_processing.data_class import Dataset
import numpy as np
import json
import os
from scripts.visualization.visuals import training_performance, error_vs_atoms

def dump_to_file(dictionary:dict,filename:str):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(dictionary,file,ensure_ascii=False, indent=4)
        file.close()

def mse_calc(x,y):
    return np.mean((x - y)**2)


def print_samples(x,y,index=10):
    for i in range(0,index):
        print(x[i],y[i])

def metrics(model_class,dataset,x,y,threshold=50):
    metrics_path = os.path.join(model_class.folder,model_class.model_name,'metrics.txt')
    with open(metrics_path,'w') as file:
        deviations = np.abs(x - y)
        sorted_indices = np.argsort(deviations)[::-1]  # Sort in descending order

        top_indices = sorted_indices[:5]
        above_threshold_indices = sorted_indices[deviations[sorted_indices] > threshold]

        file.write(f"Avg Deviation: {np.mean(deviations):.2f}\n")

        file.write(f"Errors greather than {threshold}:\n")
        for i, idx in enumerate(above_threshold_indices):
            file.write(f"#{i+1}: Deviation={deviations[idx]:.2f}, Actual={y[idx]}, Predicted={x[idx]}\n")
            file.write(f"{dataset.iloc[[idx]]}\n\n")  # Print corresponding row from dataset
        
        file.close()
        
        

def inference(dataset, model_class,cnn_model=False):

    x_test,y_test = dataset.load_data(dataset="testing")
    testing_dataframe = dataset.load_csv(dataset='testing')
    if cnn_model:
        x_test= np.expand_dims(x_test, axis=-1)
    predictions,mse=model_class.predict(x_test,y_test)
    model_path = os.path.join(model_class.folder,model_class.model_name)
    error_vs_atoms(x_test=x_test,y_test=y_test,predictions=predictions,save_to=f"{model_path}/error.png")

    metrics(model_class,testing_dataframe,predictions,y_test)
    return predictions

def train():
    dataset = Dataset(complete_dataset='data/filtered_molecules.csv',npz_folder='datasets',csv_folder='datasets')
    new_network = NeuralNetworkManager(model_name="testing",folder_name="cnn_model")
    
    x_train,y_train = dataset.load_data(dataset="training")
    x_train= np.expand_dims(x_train, axis=-1)
    
    model_path = os.path.join(new_network.folder,new_network.model_name)
    parameters = {
        'filter_list':[8,4,2],
        'kernel_size':6,
        'activation':'relu'}
    new_network.cnn_network(x_train_shape=x_train.shape,
                            filter_list=parameters['filter_list'],
                            kernel_size=parameters['kernel_size'],
                            activation=parameters['activation'])
    
    new_network.model.summary()
    hyper_parameters={
        'learning_rate':1e-4,
        'epochs':300,
        'batch_size':4,
        'call_backs':[]
    }

    new_network.compile_model(learning_rate=hyper_parameters['learning_rate'])
    new_network.fit_model(x_train,y_train,
                          epochs=hyper_parameters['epochs'],
                          batch_size=hyper_parameters['batch_size'],
                          callbacks=hyper_parameters["call_backs"])
    
    parameters.update(hyper_parameters)
    dump_to_file(parameters,f"{model_path}/parameters.json")
    predictions = inference(dataset,new_network,cnn_model=True)
    training_performance(new_network.history.history,save_to=f"{model_path}/training_loss.png")


def load():
    dataset = Dataset(complete_dataset='data/filtered_molecules.csv',npz_folder='datasets',csv_folder='datasets')
    new_network = NeuralNetworkManager(model_name='testing',folder_name="cnn_model") 
    model_path = os.path.join(new_network.folder,new_network.model_name)
    new_network.load_model(epochs=300)
    predictions = inference(dataset,new_network,cnn_model=True)


if __name__ == "__main__":
    load()