import torch

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
def learning_curve(history):
    '''
    Return the accuracy curve and loss curve
    history: the history of the model, which records the accuracy and loss of both training set and validation set 
    '''
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
    '''
    Save the model to a desinated place
    model: model you wnat to save
    pathname: path to the palce you want to save the model
    '''
    torch.save(model,pathname)
    
def load_model(pathname):
    model=torch.load(pathname)
    return model
    
def predict(model,data_loader):
    '''
    Return tensor of predicted values and tensor of ground truth labels
    model: trained model
    data_loader: data_loadet that load the data that need to be predicted
    '''
    model.eval()
    predicted=[]
    labels=[]
    for batch_idx,(img,label) in enumerate(data_loader):
        outputs=model(img)
        predicted.append(outputs)
        labels.append(label)
    predicted_t=torch.stack(predicted)
    label_t=torch.stack(labels)
    return predicted_t,label_t

def show_val_images(model,val_loader):
    '''
    Display the images, its ground truth label and its predicted label
    model: trained model
    val_loader: validation dataloader that loads the data need to be displayed
    '''
    classes_name={0: 'normal', 1: 'infected(non_covid)', 2: 'infected(covid)'}
    model.eval()
    for content in val_loader:
        img_ls = content[0] #list of image tensors in validation
        label_ls = content[1] #list of ground truth labels for each image in validation
        predicted = model(img_ls)
        # print("Predicted labels: ", predicted)
        rows=5
        cols=5
        figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(20,18) )
        for i,image in enumerate(img_ls):
            ax.ravel()[i].imshow(image[0])
            label=torch.max(label_ls[i],dim=0)[1]
            predicted_label=torch.max(predicted[i],dim=0)[1]
            title=("Ground truth label: "+classes_name[label.item()]+"\n"+
                    "Predicted label: "+classes_name[predicted_label.item()])
            ax.ravel()[i].set_title(title,fontsize=8)
            ax.ravel()[i].set_axis_off()
        
        plt.show()
        
    return


def calculate_confusion_df(model,data_loader):
    '''
    Return the confusion matrix as dataframe
    model: trained model
    data_loader: data loader that loads data
    '''

    # Initialisation
    count00=0
    count01=0
    count02=0
    count10=0
    count11=0
    count12=0
    count20=0
    count21=0
    count22=0 
    model.eval()
    
    for bath_idx,(img,label) in enumerate(data_loader):
        predicted = model(img)
        pred_max = predicted.argmax(1)
        
        #confusion matix
        label_idx = label.argmax(1)
    #print("label ls", label_ls)
    #print("predicted argmax ls", pred_max)
    #multilabel_confusion_matrix(label_ls.numpy(),predicted.numpy())
    
    #manually build confusion matrix for classes normal, infected(non-covid),infected(covid)
        for act,pred in zip(label_idx,pred_max):
            if act==torch.Tensor(1).fill_(0):
                if pred==torch.Tensor(1).fill_(0):
                    count00+=1
                elif pred==torch.Tensor(1).fill_(1):
                    count01+=1
                elif pred==torch.Tensor(1).fill_(2):
                    count02+=1
            elif act==torch.Tensor(1).fill_(1):
                if pred==torch.Tensor(1).fill_(0):
                    count10+=1
                elif pred==torch.Tensor(1).fill_(1):
                    count11+=1
                elif pred==torch.Tensor(1).fill_(2):
                    count12+=1
            elif act==torch.Tensor(1).fill_(2):
                if pred==torch.Tensor(1).fill_(0):
                    count20+=1
                elif pred==torch.Tensor(1).fill_(1):
                    count21+=1
                elif pred==torch.Tensor(1).fill_(2):
                    count22+=1

    actual_0 = [count00,count01,count02]
    actual_1 = [count10,count11,count12]
    actual_2 = [count20,count21,count22]
    total = [actual_0,actual_1,actual_2]
    conf_matrix = pd.DataFrame(total, index=['Actual Normal','Actual Infected (non_covid)','Actual Infected (covid)'],columns=['Predicted Normal','Predicted Infected (non_covid)','Predicted Infected (covid)'])
    
    return conf_matrix