from network_class import NeuralNetworkManager
from data_class import Dataset
import numpy as np
from visuals import training_performance, error_vs_atoms

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
    x_train,y_train = dataset.load_data(dataset="training")
    x_train= np.expand_dims(x_train, axis=-1)
    new_network = NeuralNetworkManager(model_name="kernel_6",folder="cnn_model")
    new_network.cnn_network(x_train_shape=x_train.shape,filter_list=[8,4,2],kernel_size=6)
    new_network.model.summary()
    new_network.compile_model(learning_rate=1e-4)
    new_network.fit_model(x_train,y_train,epochs=200,batch_size=4,callbacks=[])
    x_test,y_test = dataset.load_data(dataset="testing")
    x_test= np.expand_dims(x_test, axis=-1)
    predictions, mse = new_network.predict(x_test,y_test)
    metrics(predictions,y_test)
    training_performance(new_network.history.history,save_to=f"{new_network.folder}/{new_network.model_name}/{new_network.model_name}_training_loss.png")
    error_vs_atoms(x_test=x_test,y_test=y_test,predictions=predictions,save_to="error.png")


if __name__ == "__main__":
    main()