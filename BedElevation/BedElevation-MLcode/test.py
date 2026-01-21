from __future__ import print_function
import argparse
import torch
from torchvision.transforms import ToTensor
import os
import numpy as np
from model import ConvNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--filename', type=str, required=True, help='model file to use')
opt = parser.parse_args()

y_o = np.load('test/input/'+opt.filename)
y_o = y_o.astype(np.float32)
y_o = y_o[2,]
y_o = torch.from_numpy(y_o)
#print(y.shape)
input = y_o.view(1, 1, y_o.size()[0], y_o.size()[1])
#print(y.shape)
    
y_t = np.load('test/target/Avg_0_016000_2d.npy')
y_t = y_t.astype(np.float32)
y_t = y_t[2,]
y_t = torch.from_numpy(y_t)
#print(y.shape)
target = y_t.view(1, 1, y_t.size()[0], y_t.size()[1])

model = ConvNet().to('cpu')
checkpoint = torch.load(opt.model)
model.load_state_dict(checkpoint['model_state_dict'])
#model.eval()
   
out = model(input)

print(torch.mean(torch.mean((out-target)**2)))    
print(torch.mean(torch.mean(torch.abs(out-target))))
print(target[0,0,200,:])
print(input[0,0,200,:])
print(out[0,0,200,:])
#print(model.state_dict('layer1.0.weight'))
