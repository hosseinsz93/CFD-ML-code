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
parser.add_argument('--path', type=str, default='./test', help='where to run the model(train, test or check)')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()


def sdmkdir(x):
    if not os.path.isdir(x):
        os.makedirs(x)
def data_process(filepath,outname):
    y = np.load(filepath)
    y = y.astype(np.float32)
    y = y[[0,10,11,12],]
    y = torch.from_numpy(y)
    #print(y.shape)
    input = y.view(1, y.size()[0], y.size()[1], y.size()[2])
    #print(y.shape)

    model = ConvNet().to('cpu')
    checkpoint = torch.load(opt.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.eval()

    if opt.cuda:
        model = model.cuda()
        input = input.cuda()

    out = model(input)
    out = out.cpu()    
    out_y = out[0].detach().numpy()
    out_y = np.concatenate((out_y,out_y,out_y), axis=0)
    #print(out_y.shape)
    #out_y = out_y.reshape(y.size()[0],y.size()[1]*y.size()[2]*y.size()[3])
    #print(out_y.shape)
    #info = filepath.replace('_',' ').replace('Vrf',' ').split(' ')
    #v = [int(k) for k in info if k.isnumeric()]
    #print(v[0])
    #out_y=v[0]*out_y
    #print(out_y[2,0,0,0])
    print(out_y.shape)

  
    print('output data saved to ', outname)
    np.save(outname, out_y)

def path_process(path,out):
    
    sdmkdir(out) 
    all =[]
    for x in os.listdir(path):
        if x.endswith(".npy"):
            all.append([os.path.join(path,x),x])
    for x,name in all:
        data_process(x,os.path.join(out,name))

if __name__=='__main__':
    input_path= '/gpfs/scratch/zexizhang/Sediment/data/{}/input/'.format(opt.path)
    output_path = '{}/output'.format(opt.path)
    path_process(input_path,output_path)
