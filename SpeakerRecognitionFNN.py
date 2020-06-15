# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:19:45 2020

@author: alps
"""


import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import diff
import soundfile as sf
import sounddevice as sd
import librosa
import pandas as pd
from recorder import Recorder
import time
import tkinter 
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import pickle

def pltSignal(signal):
    Time = np.linspace(0, len(signal) / fs, num=len(signal))
    
    plt.figure(1)
    plt.title("Signal Wave...")
    plt.plot(Time, signal)
    plt.show()
    
def autoCorrelation(frame):
    ac = []
    sum1 = 0
    for i in range(len(frame)):
        sum1 = 0
        for j in range(len(frame)-i):
            s = frame[j]*frame[j+i]
            sum1 = sum1 + s
        ac.append(sum1)
    return ac[1]

def speechRecorder(speaker_name):
    
    file_name = "{}.wav".format(speaker_name)
    global running
    running = None
    root = tkinter.Tk()
    
    text = tkinter.Label(root, text = speaker_name)
    text.pack()
    
    button_rec = tkinter.Button(root, text='Start', command = lambda : start(speaker_name))
    button_rec.pack()
    
    button_stop = tkinter.Button(root, text='Stop', command=stop)
    button_stop.pack()
    
    root.mainloop() 
    
    return file_name


def start(speaker_name):
    
    global running
    file_name = "{}.wav".format(speaker_name)
    if running is not None:
        print('already running')
    else:
        running = rec.open(file_name, 'wb')
        print("ses kaydediliyor")
        running.start_recording()

def stop():
    global running
    if running is not None:
        running.stop_recording()
        print("ses kaydı kapatıldı")
        running.close()
        running = None
    else:
        print('not running')

# In this section we break data into small windows. fd equivalent of each window in seconds
# f_size is the size of the sample number and frame_count is the total number of Windows.

def voice_splitter(voice_file, fd=0.025):
    [data, fs] = sf.read(voice_file)
    data = data[:, 1]
                                 #takes only one channel of the speach signal
    maks_data = max(data)                               # takes max value to mormalize data between 0-1
    data[:] = [x / maks_data for x in data]             #narmalizes data here
    global dd
    dd = data  

    # fd = 0.025;
    f_size = round(fd * fs)                                    #defines frame size according to its sample count
    frame_count = math.floor(len(data) / f_size)                #number of frames defined here

    splitted = np.zeros((frame_count, f_size))                  #copy of the split martris
    use_split = np.zeros((frame_count, f_size))

    for i in range(frame_count):
        splitted[i, :] = data[i * f_size: (i + 1) * f_size]
        use_split[i, :] = data[i * f_size: (i + 1) * f_size]

    return splitted, use_split, f_size, fs


# This section uses zcr,Ste and Max Amplitude methods to detect audible and silent windows.
def find_voiced(splitted, use_split):
    [r, c] = splitted.shape;              # get shape of the splitted data as dimensions
    stac = np.ones(r)                      # create vector of zeros for zero crossing rate values
    STEe = np.zeros(r)                      # create vector of zeros for short term energy values
    max_split = np.zeros(r)                  # create zeros for finding values under some apmlitudes
    
    for i in range(r):
        frame = splitted[i, :]
        #stac[i] = autoCorrelation(frame)
        STEe[i] = sum(np.power(frame, 2))


    maxs_ste = max(STEe)
    STEe[:] = [x / maxs_ste for x in STEe]


    for i in range(r):
        max_split[i] = max(splitted[i, :])
        
    id = []
    for i in range(r):                                               # and max_split[no] > 0.03 according to frames STE, ZCR and max_split values program
        if (stac[i] >= 0.1 and STEe[i] >= 0.001):                              # finds voiced frames ZCRr[no] <= 0.2 and max_split[no] > 0.03 and
            id.append(i)
            
    file = open("istiklal_anne_stac","rb")
    stac = pickle.load(file)
    
    return id,stac,STEe


# Words are extracted using the audible and silent windows in this section.
def word_extracting(splitted, id, f_size, flat=True):
    # subtracting words using spaces in fragmented data when flat = False.
    # Then the longest word is determined and all the words are brought to the same length by means of zero-padding.
    # Some decals are not words, but they can be seen as words by the program.
    # Then we collect all the words in a matrix called words.
    if (flat == False):
        borders = [0]
        for i in range(len(id) - 1):
            if (np.absolute(id[i] - id[i + 1]) >= 10):
                borders.append(id[i]+4)

        max_length_word = max(diff(borders)) + 1
        words = np.zeros((len(borders) - 1, max_length_word * f_size))
        for i in range(len(borders) - 1):
            a = splitted[borders[i]:borders[i + 1], :]
            a = (a.flatten())
            difference = max_length_word * f_size - len(a)
            a = np.append(a, np.zeros(difference))
            words[i, :] = a
        # elimination
        x = 0
        for i in range(len(words)):
            x = x + len(np.trim_zeros(words[i, :]))
        global words_watch
        words_watch = words
        aver = x / len(words)
        for i in range(words.shape[0]-1,-1,-1):
            if (len((np.trim_zeros(words[i, :]))) < aver / 3):
                words = np.delete(words, i, axis=0)

        return words
    # flat=true when entered fragmented data (splitted) by making it flat
    # It's being stripped of # 0s and ready to be torn apart.
    
    else:
        borders = [0]
        for i in range(len(id) - 1):
            if (np.absolute(id[i] - id[i + 1]) >= 10):
                borders.append(id[i])


        words = []
        for i in range(len(borders) - 1):
            a = splitted[borders[i]:borders[i + 1], :]
            a = (a.flatten())
            words = np.append(words, a)

        return words


# The MFCC properties of each word or audio track are removed and rotated as 20 vectors.
def mfcc_coefs(words, fs):
    [r, c] = words.shape
    m_c = np.zeros((r, 20))
    for i in range(r):
        a = np.trim_zeros(words[i, :])
        acc = librosa.feature.mfcc(a, fs)
        [row, col] = acc.shape
        aver_mfcc = np.zeros(row)
        for no in range(row):
            aver_mfcc[no] = sum(acc[no, :]) / col
        m_c[i, :] = aver_mfcc

    return m_c


def splitter(words, labels, fd=0.25, fs=96000, flat=False):
    # If the word extraction section represents each line in the matrix, the words are not flattened
    # using the (flat=False) option, we break these words into smaller pieces.
    if (flat == False):
        out_split = []
        for z in range(words.shape[0]):
            a = words[z, :]
            a = np.trim_zeros(a)
            f_size = round(fd * fs);
            frame_count = math.floor(len(a) / f_size)
            frames = np.zeros((frame_count, f_size))
            if (z == 0):
                for_mfcc = [[0] * (f_size)]
            for x in range(frame_count):
                frames[x, :] = a[x * f_size: (x + 1) * f_size]
                out_split = np.append(out_split, [labels[z]])
            for_mfcc = np.append(for_mfcc, frames, axis=0)
        for_mfcc = np.delete(for_mfcc, (0), axis=0)
        
        return for_mfcc, out_split
    #  The flattened word matrix is broken into smaller pieces and ready for analysis.
    
    else:
        a = words
        a = np.trim_zeros(a)
        f_size = round(fd * fs);
        frame_count = math.floor(len(a) / f_size)
        frames = np.zeros((frame_count, f_size))
        for x in range(frame_count):
            frames[x, :] = a[x * f_size: (x + 1) * f_size]
        for_mfcc = frames

        return for_mfcc
    
    
rec = Recorder(channels=2)
#speaker1_file = speechRecorder("nedim")
#speaker2_file = speechRecorder("sefa")
#speech_file = speechRecorder("istik_baba_fnn")


# the first speaker's voice analysis is being done.
splitted, use_split, f_size, fs = voice_splitter("istik1.wav")     #nebahat.wav
voiced_frames,stac,ste = find_voiced(splitted, use_split)
words = word_extracting(splitted, voiced_frames, f_size, flat=True)
labels = np.full((1, words.shape[0]), 1)
for_mfcc = splitter(words, labels, flat=True)
MFcc = mfcc_coefs(for_mfcc, fs)

# the second speaker's voice analysis is being performed.
splitted2, use_split2, f_size2, fs2 = voice_splitter("istik2.wav")     #sefa2.wav
voiced_frames2,stac,ste = find_voiced(splitted2, use_split2)
words2 = word_extracting(splitted2, voiced_frames2, f_size2, flat=True)
labels2 = np.full((1, len(words2)), 0)
for_mfcc2 = splitter(words2, labels2, flat=True)
MFcc2 = mfcc_coefs(for_mfcc2, fs2)


# Removing sentences of conversation.
labels3 = [0]
splitted3, use_split3, f_size3, fs3 = voice_splitter("istik3.wav")     #ev66.wav
voiced_frames3,stac,ste = find_voiced(splitted3, use_split3)
words3 = word_extracting(splitted3, voiced_frames3, f_size3, flat=False)

# for_mfcc3,out_split = splitter(words3,labels3,flat=False)
# MFcc3 = mfcc_coefs(for_mfcc3,fs3)

# the data of the first and second speakers is shown as a table so that they are ready for machine learning.
# output string represents the output. The label of the first speaker is 1 and that of the second speaker is 0.
MFcc = pd.DataFrame(MFcc)
MFcc['output'] = 1
MFcc2 = pd.DataFrame(MFcc2)
MFcc2['output'] = 0

# obtaining datas are concataneting
raw_data = pd.concat([MFcc, MFcc2], axis=0)
# raw of the inputs are scrambelled
raw_data.sample(frac=1)

# the input and output sections are separated and ready for training.
trainy = raw_data["output"]
trainX = raw_data.drop("output", axis=1)

#speak_order = [1,0,1,0,1,0,1,0,1]             #baba istiklal
#speak_order = [0,1,0,1,0,1,0]                   #anne istiklal
testy = []

sentence_order = []
mfcc_counts_in_sentence = []
a = range(20)
testX = pd.DataFrame(columns = a)
for i in range(len(words3)):
    for_mfcc3 = splitter(words3[i], labels3, flat=True)
    MF = mfcc_coefs(for_mfcc3, fs3)
    MFcc_test = pd.DataFrame(MF)
    testX = testX.append(MFcc_test)
    for no in range(len(MFcc_test)):
        #testy.append(speak_order[i])
        sentence_order.append(i)
    mfcc_counts_in_sentence.append(len(MFcc_test))

#sent_order = pd.DataFrame({'sentence_order': sentence_order})


import torch
import torch.utils.data                      # The core package of Torch
import torch.nn as nn               # The nn package is used for building neural networks
import torch.nn.functional as F     # Contains all the functions in the torch.nn library

from sklearn.model_selection import train_test_split
#import scikitplot as skplt

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


train_all_numpy = trainX.to_numpy()
train_all_label_numpy = trainy.to_numpy()
test_numpy = testX.to_numpy()

train, validation, train_label, validation_label = train_test_split(train_all_numpy, train_all_label_numpy, test_size=0.2)


train_all_tensor = torch.as_tensor(train_all_numpy).type(torch.FloatTensor)
train_all_label_tensor = torch.as_tensor(train_all_label_numpy)
test_tensor = torch.as_tensor(test_numpy).type(torch.FloatTensor)

train_tensor = torch.as_tensor(train).type(torch.FloatTensor)
train_label = torch.as_tensor(train_label)

validation_tensor = torch.as_tensor(validation).type(torch.FloatTensor)
validation_label = torch.as_tensor(validation_label)


#testy_tensor = torch.as_tensor(testy)

class FNet(nn.Module):             
    def __init__(self):
        super(FNet, self).__init__()
                   # 784 inputs, connects to hidden layer with 600 nodes
        self.fc1 = nn.Linear(in_features=20, out_features=16)
                   # 600 nodes connects to hidden layer with 500 nodes
        self.fc2 = nn.Linear(in_features=16, out_features=10) 
                   # 500 nodes connects to hidden layer with 250 nodes
        self.fc3 = nn.Linear(in_features=10, out_features=6) 
                   # connection between the last hidden layer 
                   # and the output layer (with 10 nodes)
        self.fc4 = nn.Linear(in_features=6, out_features=3)  
                                                                
    def forward(self, x):
        x = x.view(-1,20)          # Put all the entries of the image in the vector
        x = F.relu(self.fc1(x))     # Input x into first layer and apply a ReLU
                                    # to the nodes in this layer
        x = F.relu(self.fc2(x))        
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x     


def get_accuracy(predictions, true_labels):
    _, predicted = torch.max(predictions, 1)
    corrects = (predicted == true_labels).sum()
    accuracy = 100.0 * corrects/len(true_labels)
    return accuracy.item()

def training (dataloader, epochs, model, criterion, optimizer):

    train_accuracies, train_losses = [], []
    
    # set the train mode
    model.train()
    
    # loop over training dataset multiple times
    for epoch in range(epochs):        

        train_loss = 0 
        train_accuracy = 0
        num_batch = 0
        
        # iterate over all batches
        for data, labels in dataloader:
            
            # zero the parameters gradient to not accumulate gradients from previous iteration
            optimizer.zero_grad()
            
            # put data into the model
            predictions = net(data)
            
            # calculate loss
            loss = criterion(predictions, labels)
            
            # calculate accuracy
            accurasy = get_accuracy(predictions, labels)
            
            # compute gradients
            loss.backward()
            
            # change the weights
            optimizer.step()
            
            num_batch += 1
            train_loss += loss.item()
            train_accuracy += accurasy
    
        epoch_accuracy = train_accuracy/num_batch
        epoch_loss = train_loss/num_batch        
        train_accuracies.append(epoch_accuracy)
        train_losses.append(epoch_loss)
        
        print("Epoch: {}/{} ".format(epoch + 1, epochs),
              "Training Loss: {:.4f} ".format(epoch_loss),
              "Training accuracy: {:.4f}".format(epoch_accuracy))
    
    return train_accuracies, train_losses



torch.manual_seed(0)                                  


batch_size = 2                                            # Set the batch_size
train_all_dataset = torch.utils.data.TensorDataset(train_all_tensor, train_all_label_tensor)
trainloader = torch.utils.data.DataLoader(train_all_dataset, batch_size = batch_size, shuffle = True)

epochs = 80                                               # set number of epohs

net = FNet()                                              # initialize network
criterion = nn.CrossEntropyLoss()                         # set criterion
optimizer = torch.optim.Adam(net.parameters(), lr = 3e-4) # set optimizer

                                                          # start training process
train_accuracies, train_losses = training(trainloader, epochs, net, criterion, optimizer)


net.eval()  

test_predictions = net(test_tensor)

_, test_predicted = torch.max(test_predictions.data,1)

output = pd.DataFrame({'output_fnn': test_predicted})


speaker1 = []
speaker2 = []

start = 0
for i in range(len(words3)):
    no = mfcc_counts_in_sentence[i]
    a = sum( (test_predicted[start: (start+no) ] ).numpy() )
    if ( (a / no) > 1/2 ):
        speaker1.append(words3[i])
    else:
        speaker2.append(words3[i])
    start += no 





    
# accuracy test according to handgiven labels
"""
correct = 0
false = 0
for i in range(len(testy)):
    if(test_predicted[i] == testy[i]):
        correct += 1
    else:
        false  += 1

print("accuracy : {}".format(correct/(false + correct)))
"""
    
     