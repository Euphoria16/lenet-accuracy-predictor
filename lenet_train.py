
# coding: utf-8

# # Convolutional Neural Network (CNN) for MNIST classification


import numpy as np
import torch
import time
import platform
import torch.nn.utils.prune as prune

import argparse
import torch.nn as nn
import torch.nn.functional as F
from model import QuantLeNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from MNISTtools import load, show

def normalize_MNIST_images(x):
    '''
    Args:
        x: data
    '''
    x_norm = x.astype(np.float32)
    return x_norm*2/255-1



def evaluate(net_gpu, xtest_gpu, ltest):
    y = net_gpu(xtest_gpu)
    return ((ltest == y.max(1)[1].cpu()).float().mean())
def train_and_eval(x_train, label_train, model, x_test, label_test, total_epochs, weight_bit_num, batch_size=100, lr=.001, lr_momentum=.9):
    '''
    
    Args:
        x_train: training samples
        label_train: testing samples
        model: neural network
        total_epochs: number of epochs
        batch_size: minibatch size
        lr: step size
        lr_momentum: momentum
    '''
    N = x_train.size()[0]
    num_batch = N // batch_size
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=lr_momentum)
    best_acc=0
    for epoch in range(total_epochs):
        running_loss = 0.0
        shuffled_indices = np.random.permutation(num_batch)
        for k in range(num_batch):

            minibatch_indices = range(shuffled_indices[k] * batch_size, (shuffled_indices[k] + 1) * batch_size)
            inputs = x_train[minibatch_indices]
            labels = label_train[minibatch_indices]


            optimizer.zero_grad()


            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
            # if k % 100 == 99:
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, k + 1, running_loss / 100))
                running_loss = 0.0
        acc=evaluate(model, x_test, label_test)
        if acc>best_acc:
            best_acc=acc
            state_dict=model.state_dict()
            torch.save(state_dict,'weight_bit_'+str(weight_bit_num)+'_best.pth')
        print('epoch{}:'.format(epoch),acc)

parser = argparse.ArgumentParser()
parser.add_argument('--weight_bit', default=0, type=int,)
args=parser.parse_args()
weight_bit_width=args.weight_bit
# start = time.time()

xtrain, ltrain = load(dataset='training', path='dataset/')
xtest, ytest = load(dataset='testing', path='dataset/')

xtrain = normalize_MNIST_images(xtrain)
xtest = normalize_MNIST_images(xtest)


xtrain = xtrain.reshape([28,28,-1])[:,:,None,:]
xtest = xtest.reshape([28,28,-1])[:,:,None,:]

# moveaxis
xtrain = np.moveaxis(xtrain, (2,3), (1,0))
xtest = np.moveaxis(xtest, (2,3), (1,0))

xtrain = torch.from_numpy(xtrain)
ltrain = torch.from_numpy(ltrain)
xtest = torch.from_numpy(xtest)
ytest = torch.from_numpy(ytest)

xtrain_gpu = xtrain.to(device)
ytrain_gpu = ltrain.to(device)
xtest_gpu = xtest.to(device)
ltest_gpu = ytest.to(device)

weight_bit_width_list=[2,3]#,4,5,6,8,16]#7*16
vector_list=[]
acc_list=[]

for i in range(weight_bit_width, weight_bit_width + 1):

    for amount_num in (range(15,-1,-8)):


        net_gpu = QuantLeNet(weight_bit_width=weight_bit_width_list[i]).to(device)
        #save the best model on validation set

        train_and_eval(xtrain_gpu, ytrain_gpu, net_gpu, xtest_gpu, ytest, total_epochs=2, weight_bit_num=weight_bit_width)
        # end = time.time()
        # print(f'It takes {end-start:.6f} seconds.')

        net_gpu.load_state_dict(torch.load('weight_bit_' + str(weight_bit_width) + '_best.pth'))
        net_gpu.to(device)

        resume_acc = evaluate(net_gpu, xtest_gpu, ytest)
        print('quantization best accuracy: {:.5f}'.format(resume_acc))

        #prune
        conv_module = net_gpu.conv2
        prune.ln_structured(conv_module, name='weight', amount=amount_num, n=2, dim=0)

        # print(list(model.conv2.named_parameters()))
        #
        prune_acc = evaluate(net_gpu, xtest_gpu, ytest)

        print('prune accuracy: {:.5f}'.format(prune_acc))


        fine_tune_epoch=2
        train_and_eval(xtrain_gpu, ytrain_gpu, net_gpu, xtest_gpu, ytest, total_epochs=fine_tune_epoch, weight_bit_num=weight_bit_width)


        net_gpu.load_state_dict(torch.load('weight_bit_' + str(weight_bit_width) + '_best.pth'))
        net_gpu.to(device)
        resume_fine_acc = evaluate(net_gpu, xtest_gpu, ytest)
        print('finetune accuracy after pruning: {:.5f}'.format(resume_fine_acc))

        #generate one-hot label
        weight_bit_vector = np.zeros(16)
        prune_vector = np.zeros(len(range(15,-1,-1)))
        weight_bit_vector[i-1] = 1
        prune_vector[amount_num] = 1

        vector = (np.hstack((weight_bit_vector, prune_vector)))
        vector_list.append(vector)
        acc_list.append(resume_fine_acc)

np.save('weight_bit_' + str(weight_bit_width) + '.npy', np.array(vector_list))
np.save('acc_' + str(weight_bit_width) + '.npy', np.array(acc_list))
print('-----finished sampling------')