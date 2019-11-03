#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 18:42:21 2019

@author: siggy
"""
import torch
import torch.nn as nn
import numpy as np
import random
import os
import cv2
from PIL import Image

class Image_Processer(nn.Module):
    def __init__(self):
        super(Image_Processer, self).__init__()
        
        self.conv1 = nn.Conv2d(9, 1, kernel_size=9, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=6, stride=2)
        self.linear1    = nn.Linear(954, 500)
        self.linear2    = nn.Linear(500, 100)
        self.linear3    = nn.Linear(100,50)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        #print(x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.relu(self.linear3(x))    
        return x
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.processor1 = Image_Processer()
        self.processor2 = Image_Processer()
        self.processor3 = Image_Processer()
        
        self.gru = nn.GRU(input_size=26700, hidden_size=1000, batch_first=True)
        
        self.linear1 = nn.Linear(1000, 100)
        self.linear2 = nn.Linear(100,20)
        self.linear3 = nn.Linear(20, 4)
        
    def forward(self, inputs):
        x = inputs[0]
        y = inputs[1]
        z = inputs[2]
        
        #print("Input X shape:", x.shape)
        x = self.processor1(x)
        x = x.view((-1, 1, 534*50))
        y = self.processor2(y)
        y = y.view((-1, 1, 534*50))
        z = self.processor3(z)
        z = z.view((-1, 1, 534*50))        
        
        seq = torch.cat((x, y, z), dim=1)
        
        output, h_n = self.gru(seq)
        
        out = self.linear1(h_n)
        out = self.linear2(out)
        out = self.linear3(out)
        
        return out
def test1():
    model = Image_Processer()
    image = "./Anger-features/flickr-5.npy"
    arr = np.load(image)
    print(arr.shape)
    
    arr = torch.from_numpy(arr)
    arr1 = arr[:3]
    arr2 = arr[3:6]
    arr3 = arr[6:]
    
    arr11 = arr1[0]
    for a in arr1[1:]:
        arr11 = np.concatenate((arr11, a), axis = 2)
    
    arr11 = torch.from_numpy(arr11)
    print(arr11.shape)
    arr11 = arr11.view((1, 9, 1080, 1920)).float()
    print(arr11.shape)
    
    out = model(arr11)
    out = out.view((-1, 1, 534*50))
    print(out.shape) 

def train_model(epochs, batch_size):
    model = Classifier()
    objective = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    example_size = 557
    folders = [
            "Anger-features",
            "Fear-features",
            "Joy-features",
            "Sadness-features"]   
    epoch_loss = 99999
    for e in range(epochs):
        print("Beginning Epoch:",e)
        losses = list()
        for minibatch in range(example_size//batch_size):
            optimizer.zero_grad()
            current_files = list()
            current_truths = list()
            chosen_list = list()    
            i = 0
            while i < batch_size:
        
                curr_emotion = random.choice(folders)
                files = os.listdir(curr_emotion)
                check = 0
                while check < 1:
                    chosen = random.choice(files)
                    if ".txt" not in chosen:
                        check += 1
                emotion_index = folders.index(curr_emotion)
                #print(chosen)
                path = curr_emotion + "/" + chosen
                #print(path)
                if path not in chosen_list:
                    f = np.load(path, allow_pickle=True)
                   # print(f.shape)
                    if f.shape[0] == 9:
                        #print("F shape: ",f.shape)
                        
                        f1 = f[:3]
                        f2 = f[3:6]
                        f3 = f[6:] 
                        processed = list()
                        for fv in [f1, f2, f3]:  
                            f11 = fv[0]
                            for a in fv[1:]:
                                f11 = np.concatenate((f11, a), axis = 2)   
                                #print("f11 shape: ", f11.shape)
                            processed.append(f11)
                        
                        
                        current_files.append(np.array(processed))
                        #print(np.array(processed).shape)
                        current_truths.append(emotion_index)
                        i+=1
            
            current_files = np.array(current_files)
            current_files = torch.from_numpy(current_files).float()
            #print(current_files.shape)
            #current_files = current_files.view(3, 16, 9, 1080, 1920)
            current_files = current_files.permute(1, 0, 4, 2, 3)
            #print(current_files.shape)
            current_truths = torch.Tensor(current_truths).long()
            outcome = model(current_files).view((16, 4))
            #print(outcome.shape)
            #print(current_truths.shape)
            
            loss = objective(outcome, current_truths)
            print(float(loss))
            losses.append(float(loss))
            loss.backward()
            optimizer.step()
        if np.average(losses) < epoch_loss:
            info = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": np.average(losses),
                    "epoch": e}
            torch.save(info, "./models/model1.pth")
        
            #print(current_files[1][0].shape)
            
            #print(current_files[0][:3].shape)
        

def predict_emotion(video_file):
    emotions = ["Anger",
                "Fear",
                "Joy",
                "Sadness"]
    model = Classifier()
    info = torch.load("./models/model1.pth")
    m    = info["model_state"]
    model.load_state_dict(m)
    model.eval()
    
    
    cap = cv2.VideoCapture(video_file)    
    num_frames = cap.get(7)
    interval = num_frames // 9
    all_frames = list()
    while(cap.isOpened()):
        tryId = cap.get(1)
        #print(tr
        if cap.isOpened():
            ret, frame = cap.read()
            if frame is not None:
                pass
        else:
            break
        if (ret != True):
            break
        if tryId % interval == 0:
            #print(tryId)
            if len(all_frames) < 9 and frame is not None:
                #t1 = time.time()
                frame = np.uint8(frame)
                frame = Image.fromarray(frame)
                frame = frame.resize((1920, 1080))
                frame = np.array(frame)
                #print(frame.shape)
                #print(time.time()-t1)                        
                
                
                all_frames.append(frame) 
        
    all_frames = np.array(all_frames)  
    inp = torch.from_numpy(all_frames).view(1, 9, 1080, 1920, 3)
    inp = inp.permute(4, 0, 1, 2, 3)
    print(inp.shape)    
        
    prediction = model(inp.float()).tolist()
    idx = prediction.index(max(prediction))
    return emotions[idx]



        
    
if __name__ == "__main__":
    #bs = 16
    #train_model(10, bs)
    #test1()
    
    pred = predict_emotion("./VideoEmotionDataset4-Fear/flickr/2401021188_38d4114a26_700.mp4")
    print(pred)
 
    















    
