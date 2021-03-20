import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import time
import matplotlib.pyplot as plt

def calculate_accuracy(predicted_labels,target_labels):
    equality = torch.eq(target_labels,predicted_labels).float()
    accuracy = torch.mean(equality)
    return accuracy

def validation(model, validation_loader, criterion,binaryclassifier=False,secondclassifier=False):
    validation_loss = 0
    accuracy = 0
    newlabels=[]
    newim=[]
    for images, labels in validation_loader:
        if binaryclassifier:
            outputs = model(images)
            # get new labels
            n_labels=torch.max(labels,1)[1]
            for i in n_labels:
                if i==2:
                    i-=1    # change to class 1 (infected)
            validation_loss += criterion(outputs,n_labels).item()
            accuracy += calculate_accuracy(torch.max(outputs,1)[1],n_labels)
        elif secondclassifier:
            # only need infected images
            for i in range(len(labels)):
                if labels[i][0]!=1:
                    newlabels.append(labels[i][1:])
                    newim.append(images[i])
            # convert list to tensor 
            n_labels=torch.max(torch.stack(newlabels),1)[1]
            newimTensor = torch.stack(newim)
            outputs=model(newimTensor)
            validation_loss += criterion(outputs,n_labels).item()
            accuracy += calculate_accuracy(torch.max(outputs,1)[1],n_labels)

        else:
            outputs = model(images)
            validation_loss += criterion(outputs, torch.max(labels,1)[1]).item()
            accuracy += calculate_accuracy(torch.max(outputs,1)[1],torch.max(labels,1)[1])

    return validation_loss, accuracy

def plotfigure(acc_list,val_acc_list,losslst_train,losslst_val):
    fig,axs=plt.subplots(2)
    axs[0].plot(acc_list,label='train acc')
    axs[0].plot(val_acc_list,label='val acc')
    axs[1].plot(losslst_train,label='train loss')
    axs[1].plot(losslst_val,label='val loss')
    plt.legend()
    plt.show()

class Block(nn.Module):
    def __init__(self,in_channel,out_channel,dropout_rate):
        super(Block,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        
        self.layer2=nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.dropout(out)

        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        # for binary classifier
        self.fc1=nn.Linear(16*72*72,2)
        self.drop=nn.Dropout(0.5)
        self.fl=nn.LogSoftmax(dim=1)

    def forward(self,x):
        out=self.conv1(x)
        # print("After conv1",out.shape)
        out=out.reshape(out.size(0), -1)
        out=self.fc1(out)
        out=self.drop(out) 
        out=self.fl(out)

        return out

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        # self.layer2=nn.Sequential(
        #     nn.Conv2d(32,16,kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3,stride=2)
        # )
        self.blk1=Block(32,16,0.5)
        self.fc1=nn.Linear(16*72*72,2)
        # self.fc1=nn.Linear(16*33*33,2)
        self.drop=nn.Dropout(0.5)
        self.fl=nn.LogSoftmax(dim=1)

    def forward(self,x):
        out=self.conv1(x)
        # print("After conv1",out.shape)
        out=self.blk1(out)
        # print("After conv2",out.shape)
        out=out.reshape(out.size(0), -1)
        out=self.fc1(out)
        out=self.drop(out)
        out=self.fl(out)
        return out

def train_binary_classifier_model1(train_loader,epochs,loss_function, gen,test_loader):
    model=Classifier()

    criterion=loss_function
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    train_loss_list=[]
    train_accuracy_list=[]

    acc_list=[]
    val_acc_list=[]

    losslst_train=[]
    losslst_val=[]
    
    first_classifier_output = []
    seed = 2809
    # print('Seeding with {}'.format(seed))
    #torch.manual_seed(seed)
    gen.manual_seed(seed)
    start_time=time.time()
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images_data, target_labels) in enumerate(train_loader):
            # print(batch_idx, '----------------------')
            # get new target_labels
            if batch_idx==5:
                print('checking if data is shuffled but same',images_data[1],target_labels[1],'\n','='*20)
            target_idx_labels=torch.max(target_labels,1)[1]
            for i in target_idx_labels:
                if i==2:
                    i-=1    # change to class 1 (infected)

            outputs=model(images_data)
            loss=criterion(outputs,target_idx_labels)
            train_loss_list.append(loss.item())
            # get the predictions of infected and store in a list
            maxProb_idx = torch.max(outputs,1)[1]
            # for each epoch, the output is saved as a list
            if batch_idx==0:
                first_classifier_output.append([maxProb_idx])
            else:
                first_classifier_output[epoch].append(maxProb_idx)
            accuracy=calculate_accuracy(maxProb_idx,target_idx_labels)
            train_accuracy_list.append(accuracy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1)%10==0:
                model.eval()
                with torch.no_grad():
                    validation_loss,validation_accuracy=validation(model,test_loader,criterion=loss_function, binaryclassifier=True)

                print("Epoch: {}/{} @ {} ".format(epoch+1, epochs,time.time()),
                      "\n",
                      "Training Loss: {:.3f} - ".format(sum(train_loss_list)/len(train_loss_list)),
                      "Training Accuracy: {:.3f} - ".format(sum(train_accuracy_list)/len(train_accuracy_list)),
                      "Validation Loss: {:.3f} - ".format(validation_loss/len(test_loader)),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy/len(test_loader)))
                
                acc_list.append(sum(train_accuracy_list)/len(train_accuracy_list))
                val_acc_list.append(validation_accuracy/len(test_loader))
                losslst_train.append(sum(train_loss_list)/len(train_loss_list))
                losslst_val.append(validation_accuracy/len(test_loader))

                train_loss_list=[]
                train_accuracy_list=[]

                model.train()
    plotfigure(acc_list,val_acc_list,losslst_train,losslst_val)

    print("Training finished for the 1st classifier!","\n","Run time: {:.3f} mins".format((time.time() - start_time)/60))
    return model,first_classifier_output

def train_binary_classifier_model2(train_loader,epochs,loss_function,first_classifier_output, gen,test_loader):
    model=Classifier2()

    criterion=loss_function
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    train_loss_list=[]
    train_accuracy_list=[]

    acc_list=[]
    val_acc_list=[]

    losslst_train=[]
    losslst_val=[]

    start_time=time.time()
    seed = 2809
    # print('Seeding with {}'.format(seed))
    #torch.manual_seed(seed)
    gen.manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images_data, target_labels) in enumerate(train_loader):
            imageForTraining=[]
            labelForTraining=[]
            # print(batch_idx, '----------------------')
            if batch_idx==5:
                print('checking if data is shuffled but same',images_data[1],target_labels[1],'\n','='*20)
            # print("image data ",images_data)
            for i in range(len(images_data)):
                # check if its predicted as infected from 1st classifier
                if first_classifier_output[epoch][batch_idx][i]==1:
                    # save the image for training
                    imageForTraining.append(images_data[i])    
                    labelForTraining.append(target_labels[i][1:])
            if len(imageForTraining)!=0:
                imtensor = torch.stack(imageForTraining)
                labeltensor = torch.stack(labelForTraining)
            indiceslabel=torch.max(labeltensor,1)[1]

            outputs=model(imtensor)
            loss=criterion(outputs,indiceslabel)
            train_loss_list.append(loss.item())
            maxProb_idx = torch.max(outputs,1)[1]
            accuracy=calculate_accuracy(maxProb_idx,indiceslabel)
            train_accuracy_list.append(accuracy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1)%10==0:
                model.eval()
                with torch.no_grad():
                    validation_loss,validation_accuracy=validation(model,test_loader,criterion=loss_function,secondclassifier=True)

                print("Epoch: {}/{} @ {} ".format(epoch+1, epochs,time.time()),
                      "\n",
                      "Training Loss: {:.3f} - ".format(sum(train_loss_list)/len(train_loss_list)),
                      "Training Accuracy: {:.3f} - ".format(sum(train_accuracy_list)/len(train_accuracy_list)),
                      "Validation Loss: {:.3f} - ".format(validation_loss/len(test_loader)),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy/len(test_loader)))
                
                acc_list.append(sum(train_accuracy_list)/len(train_accuracy_list))
                val_acc_list.append(validation_accuracy/len(test_loader))
                losslst_train.append(sum(train_loss_list)/len(train_loss_list))
                losslst_val.append(validation_accuracy/len(test_loader))

                
                train_loss_list=[]
                train_accuracy_list=[]

                model.train()
    plotfigure(acc_list,val_acc_list,losslst_train,losslst_val)

    print("Training finished for 2nd classifier!","\n","Run time: {:.3f} mins".format((time.time() - start_time)/60))
    return model
