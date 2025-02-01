from network_class import NeuralNetworkManager
from data_class import Dataset
import numpy as np
import json
import os
from visuals import training_performance, error_vs_atoms

def dump_to_file(dictionary:dict,filename:str):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(dictionary,file,ensure_ascii=False, indent=4)
        file.close()

def mse_calc(x,y):
    return np.mean((x - y)**2)


def print_samples(x,y,index=10):
    for i in range(0,index):
        print(x[i],y[i])

def metrics(x,y):
    deviations = np.abs(x-y)
    max_index = np.where(deviations == np.max(deviations))[0]
    min_index = np.where(deviations == np.min(deviations))[0]
    print("Max=",np.max(deviations), "actual=",y[max_index], "predictions",x[max_index])
    print("Min=",np.min(deviations),  "actual=",y[min_index], "predictions",x[min_index])
    print("Avg=",np.mean(deviations))

def inference(dataset, model_class,cnn_model=False):
    if cnn_model:
        x_test= np.expand_dims(x_test, axis=-1)
    x_test,y_test = dataset.load_data(dataset="testing")
    predictions,mse=model_class.predict(x_test,y_test)
    print_samples(predictions, y_test)
    print("Mse=",mse)
    metrics(predictions,y_test)
    return predictions

def main():
    dataset = Dataset(complete_dataset="complete_dataset.npz",training_set="training_set.npz",testing_set="testing_set.npz")
    new_network = NeuralNetworkManager(model_name="parameter",folder="cnn_model")
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
    x_test,y_test = dataset.load_data(dataset="testing")
    x_test= np.expand_dims(x_test, axis=-1)
    predictions, mse = new_network.predict(x_test,y_test)
    metrics(predictions,y_test)
    training_performance(new_network.history.history,save_to=f"{model_path}/training_loss.png")
    error_vs_atoms(x_test=x_test,y_test=y_test,predictions=predictions,save_to=f"{model_path}/error.png")


if __name__ == "__main__":
    main()