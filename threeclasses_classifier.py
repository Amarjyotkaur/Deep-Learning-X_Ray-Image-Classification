import torch
import torch.nn as nn
import torch.optim as optim

import time
import datetime

from small_functions import save_model

class Block(nn.Module):
    '''
    A neural convolutional block that has two big layers, 
    each big layer has one convolutional layer, bath normalisation and an activation funtion ReLU().
    A drop out layer as the last layer of this block.
    '''
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


class Classifier3(nn.Module):
    '''
    Three classes classifier
    convolutional layer + fully connected layer + drop out + log softmax layer
    history: record the accuracy and loss of training set and validation set
    
    Output log(probabilities of classifying images to each class) values.
    '''
    def __init__(self):
        super(Classifier3,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        # self.blk1=Block(32,16,0.2)
        # self.blk2=Block(64,128,0.2)
        # self.blk3=Block(128,256,0.2)
        # self.blk4=Block(256,512,0.2)

        # self.avgpool=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.fc1=nn.Linear(32*72*72,3)
        self.drop=nn.Dropout(0.5)
        # self.fc2=nn.Linear(256,3)
        self.fl=nn.LogSoftmax(dim=1)

        self.history={'train_accuracy':[],'train_loss':[],'validation_accuracy':[],'validation_loss':[]}

    def forward(self,x):
        out=self.conv1(x)

        # out=self.blk1(out)
        # out=self.blk2(out)
        # out=self.blk3(out)
        # out=self.blk4(out)

        # out=self.avgpool(out)
        out=out.reshape(out.size(0), -1)
        out=self.fc1(out)
        out=self.drop(out)
        # out=self.fc2(out)   
        out=self.fl(out)

        return out

def calculate3_accuracy(predicted_labels,target_labels):
    '''
    Return the accuracy
    predicted_labels: the predited values of images
    target_labels: ground truth labels of images
    '''

    maxProb_idx = torch.max(predicted_labels,1)[1]
    
    # 
    equality = torch.eq(torch.max(target_labels,1)[1],maxProb_idx).float()
    # print('maxprob',maxProb_idx)
    # print("equality  ",equality)
    accuracy = torch.mean(equality)

    # print("predicted label  ",predicted_labels)
    # print("tarfet leable  ",target_labels)
    # print("accuracy   ",accuracy)
    return accuracy

def validation(model, validation_loader, criterion):
    validation_loss = 0
    accuracy = 0
    
    for images, labels in validation_loader:
                
        outputs = model(images)
        validation_loss += criterion(outputs, torch.max(labels,1)[1]).item()
        
        accuracy += calculate3_accuracy(outputs,labels)


    return validation_loss, accuracy

def train_model(model,train_loader,validation_loader,epochs,loss_function):
    criterion=loss_function
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    train_loss_list=[]
    train_accuracy_list=[]

    start_time=time.time()
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images_data, target_labels) in enumerate(train_loader):
            # print(batch_idx, '----------------------')
            target_idx_labels=torch.max(target_labels,1)[1]
            # print(target_idx_labels)
            # print("image data ",images_data)
            outputs=model(images_data)
            # print("output   ",outputs)
            
            loss=criterion(outputs,target_idx_labels)
            train_loss_list.append(loss.item())
            accuracy=calculate3_accuracy(outputs,target_labels)
            train_accuracy_list.append(accuracy)
            # print(train_accuracy_list)
            # print("loss ", training_loss_list)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            

            if (batch_idx+1)%10==0:
                model.eval()
                with torch.no_grad():
                    validation_loss,validation_accuracy=validation(model,validation_loader,criterion=loss_function)

                trainloss=sum(train_loss_list)/len(train_loss_list)
                valloss=validation_loss/len(validation_loader)
                trainacc=sum(train_accuracy_list)/len(train_accuracy_list)
                valacc=validation_accuracy/len(validation_loader)

                print("Epoch: {}/{} @ {} ".format(epoch+1, epochs,str(datetime.datetime.now())),
                      "\n",
                      "Training Loss: {:.3f} - ".format(trainloss),
                      "Training Accuracy: {:.3f} - ".format(trainacc),
                      "Validation Loss: {:.3f} - ".format(valloss),
                      "Validation Accuracy: {:.3f}".format(valacc))
                
                model.history['train_accuracy'].append(trainacc)
                model.history['validation_accuracy'].append(valacc)
                model.history['train_loss'].append(trainloss)
                model.history['validation_loss'].append(valloss)

                
                train_loss_list=[]
                train_accuracy_list=[]

                model.train()

            # Save model as intermediate.pt regularly 
            if batch_idx%40==0:
                save_model(model,'./model/intermediate.pt')


    print("Training finished!","\n","Run time: {:.3f} mins".format((time.time() - start_time)/60))
    return model