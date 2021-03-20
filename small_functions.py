import torch

import numpy as np 
import matplotlib.pyplot as plt
def learning_curve(history):
    #train and val loss over epochs
    fig_loss,ax_loss=plt.subplots()
    loss_train = history['train_loss']
    loss_val = history['validation_loss']
    ax_loss.plot(np.arange(1,len(loss_train)+1), loss_train, 'g', label='Training loss')
    ax_loss.plot(np.arange(1,len(loss_val)+1), loss_val, 'b', label='validation loss')
    ax_loss.set_title('Training and Validation loss')
    ax_loss.set_xlabel('Eevry 10 batches')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    plt.show()  
    
    #val and test accuracy over epochs
    fig_acc,ax_acc=plt.subplots()
    acc_train=history['train_accuracy']
    acc_val=history['validation_accuracy']
    ax_acc.plot(np.arange(1,len(loss_train)+1),acc_train)
    ax_acc.plot(np.arange(1,len(acc_val)+1),acc_val)
    ax_acc.set_title('model accuracy')
    ax_acc.set_ylabel('accuracy')
    ax_acc.set_xlabel('Eevry 10 batches')
    ax_acc.legend(['train', 'validation'], loc='upper left')
    plt.show()

    return fig_loss,fig_acc

def save_model(model, pathname):
    torch.save(model,pathname)
    
def load_model(pathname):
    model=torch.load(pathname)
    return model
    