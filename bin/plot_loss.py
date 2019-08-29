import context
import numpy as np
import matplotlib.pyplot as plt
from src.networks.cnn import CNN


NB_EPOCHS = 5

def eval_metric(name, version, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric.
    Training and validation metric are plotted in a
    line chart for each epoch.

    Parameters:
        name : name of the model
        version : version of the model
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    cnn = CNN(load_models=name, version=version)
    history = cnn.load_losses(fname=name)

    metric = history[metric_name]
    val_metric = history['val_' + metric_name]
    print(np.shape(metric)[0])
    #e = range(1, NB_EPOCHS + 1)

    plt.plot(metric, 'bo', label='Train ' + metric_name)
    plt.plot(val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    name = 'cnn-Model2_test15'
    version = 0
    eval_metric(name, version, 'loss')