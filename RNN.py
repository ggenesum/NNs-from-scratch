import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import random
from math import log
import copy
import string

#with open("text.txt", 'r', encoding="utf8", errors='ignore') as ds:
    #dataset = ds.read()

dataset = 'tesa' * 100000
#charset = [c for c in string.printable][:-6]
charset = ['t','e','s','a','b']

timesteps = []
for i in range(len(dataset)-1):
    if dataset[i]!= 'a':
        timesteps.append((dataset[i],dataset[i+1]))

def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def dsigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def Stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def Cross_entropy(prediction, target): #FIX THIS
    ce = -np.sum(target*np.log(prediction+1e-200))/10.0
    return ce

Sigmoid = np.vectorize(sigmoid)
dSigmoid = np.vectorize(dsigmoid)

class RNNlayer:
    def __init__(self, InSize, OutSize, HiddenUnits):
        #init matricies to store weights, biases, and Output values
        self.Wh = np.random.rand(HiddenUnits,HiddenUnits)*np.sqrt(2/InSize)
        self.Wx = np.random.rand(HiddenUnits,InSize)*np.sqrt(2/InSize)
        self.Ba = np.zeros((HiddenUnits,1))
        self.Wy = np.random.rand(OutSize,HiddenUnits)*np.sqrt(2/InSize)
        self.By = np.zeros((OutSize,1))
        self.Y = np.zeros((OutSize,1))
        self.A = np.zeros((HiddenUnits,1))
        self.H = np.zeros((HiddenUnits,1))
    def resmem(self):
        #the H vector is the memory of the layer
        self.H = np.zeros(self.H.shape)
    def feedforward(self,X):
        self.A = np.matmul(self.Wx,X) + np.matmul(self.Wh,self.H) + self.Ba #self.H is H at time t-1
        self.H = Sigmoid(self.A)
        self.Y = Stable_softmax(np.matmul(self.Wy,self.H)+self.By) #self.H is H at time t


def char2vect(c): #char to onehot vector
    return np.asarray([1 if charset[i] == c else 0 for i in range(len(charset))]).reshape(len(charset),1)

class Training:
    def train(self,epochs,seq_len,B = 0.9,lr=0.1,RMSProp=False):
        loss = []
        rnn = RNNlayer(5,5,40)
        display = True

        for i in range(epochs): #train
            rnn.resmem()

            LossError = 0 #not int but matrices
            WhUpdate = 0
            WxUpdate = 0
            BaUpdate = 0
            WyUpdate = 0
            ByUpdate = 0

            Swh = np.zeros(rnn.Wh.shape)
            Swx = np.zeros(rnn.Wx.shape)
            Sba = np.zeros(rnn.Ba.shape)
            Swy = np.zeros(rnn.Wy.shape)
            Sby = np.zeros(rnn.By.shape)

            LossErrors = []
            Att = [] #store values of rnn.A as we use them in BPTT
            Htt = []
            Inputs = []

            Att.append(rnn.A)

            for j in range(seq_len):
                (data, next) = timesteps[i*seq_len + j]
                (data, next) = (char2vect(data),char2vect(next))
                Inputs.append((data, next))

                rnn.feedforward(data)
                #loss.append(Cross_entropy(rnn.Y,next))

                #Compute Loss Derivative with respect to Y
                LossErrors.append(rnn.Y - next) #output error for softmax + cross entropy, many2many (compute loss at each timestep)
                Att.append(rnn.A)
                Htt.append(rnn.H)


            for j in range(len(LossErrors)):
                d_L_d_h = np.matmul(rnn.Wy.T,LossErrors[j])
                #accumulate gradients
                WyUpdate += LossErrors[j]*Htt[j].T
                ByUpdate += LossErrors[j]

                for i in reversed(range(j)):
                    temp = dSigmoid(Att[i+1])*d_L_d_h
                    WhUpdate += np.matmul(temp,Att[i].T)
                    WxUpdate += np.matmul(temp,Inputs[i][0].T)
                    BaUpdate += temp
                    d_L_d_h = np.matmul(rnn.Wh,temp)

            for d in [WhUpdate, WyUpdate, ByUpdate, WxUpdate, BaUpdate]:
                np.clip(d, -1, 1, out=d) #clip for exploding grasdients

            if display and 0 in WhUpdate:
                print("vanishing gradient...")
                display = False

            if RMSProp:
                    #RMSProp
                        #Compute Swi,Sbi
                Swh = B * Swh + (1-B) * WhUpdate*WhUpdate
                Swx = B * Swx + (1-B) * WxUpdate*WxUpdate
                Sba = B * Sba + (1-B) * BaUpdate*BaUpdate
                Swy = B * Swy + (1-B) * WyUpdate*WyUpdate
                Sby = B * Sby + (1-B) * ByUpdate*ByUpdate

                #clip for vanishig gradient,
                #this allows calculus with RMSProp but does not fix the problem
                for Swi in [Swh, Wsx, Sba, Swy, Sby]:
                    if i%2==0:
                        Swi += 1e-300
                    else:
                        Swi -= 1e-300

                rnn.Wh -= lr*WhUpdate/np.sqrt(Swh)
                rnn.Wx -= lr*WxUpdate/np.sqrt(Swx)
                rnn.Ba -= lr*BaUpdate/np.sqrt(Sba)
                rnn.Wy -= lr*WyUpdate/np.sqrt(Swy)
                rnn.By -= lr*ByUpdate/np.sqrt(Sby)

            else:
                    #SGD
                rnn.Wh -= lr*WhUpdate
                rnn.Wx -= lr*WxUpdate
                rnn.Ba -= lr*BaUpdate
                rnn.Wy -= lr*WyUpdate
                rnn.By -= lr*ByUpdate

            #learning rate decay
            #lr *= (1. / (1. + 0.000001))

        Testsuccess = 0
        Testfails = 0

        for i in range(epochs,epochs + 500):
            rnn.resmem()
            for j in range(seq_len):
                (data, next) = timesteps[i*seq_len + j]
                (data, next) = (char2vect(data),char2vect(next))
                rnn.feedforward(data)
                if np.argmax(rnn.Y)==np.argmax(next):
                    Testsuccess += 1
                else:
                    Testfails += 1

        print(Testsuccess/(Testsuccess + Testfails))

        c = 't'
        rnn.resmem()
        for i in range(seq_len):
            print(c, end ='')
            rnn.feedforward(char2vect(c))
            c = charset[np.argmax(rnn.Y,axis=0)[0]]
        print(c)

        #plt.plot(loss)
        #plt.show()
lrs = [10,5,2,1,2e-1,1e-1,1e-2,1e-3,1e-4]
for lr in lrs:
    print("\nlr = " + str(lr) +":")
    Training().train(1000,
    RMSProp = False,
    lr=lr,
    B=0.7,
    seq_len = 3) #7000 -> 0.87
