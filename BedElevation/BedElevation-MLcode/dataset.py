import torch.utils.data as data
import numpy as np
import torch
import random

from os import listdir
from os.path import join

import re


def is_data_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def load_data(filepath):
    y = np.load(filepath)
    y = y.astype(np.float32)
    y = torch.from_numpy(y)#.cuda(non_blocking=True)
#    y = y.view(y.shape[0], y.shape[1], y.shape[2], y.shape[3], y.shape[4])  
    #print(y.shape)
    return y
    
def load_data2(filepath):
    y = np.load(filepath)
    y = y.astype(np.float32)
    y = torch.from_numpy(y)#.cuda(non_blocking=True)
#    y = y.view(y.shape[0], y.shape[1], y.shape[2], y.shape[3])  
    #print(y.shape)
    return y

class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir_input, data_dir_target, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.data_filenames_input = [join(data_dir_input, x) for x in listdir(data_dir_input) if is_data_file(x)]
        self.data_filenames_target1 = [join(data_dir_target, x) for x in listdir(data_dir_target) if is_data_file(x)]
        #self.data_filenames_target2 = [join(data_dir_target2, x) for x in listdir(data_dir_target2) if is_data_file(x)]
        self.data_filenames_input=sorted(self.data_filenames_input)
        #print(self.data_filenames_input)
        self.data_filenames_target1=sorted(self.data_filenames_target1)
        #self.data_filenames_target2=sorted(self.data_filenames_target2)
        temp_target1 = self.data_filenames_target1
        #temp_target2 = self.data_filenames_target2
        
        # pair targets with inputs
        temp_target = []
        for str1 in self.data_filenames_input:
            a = str1.split('/')[-1]
            a = re.split('[_ , .]', a)
#            print(a)
            for str2 in self.data_filenames_target1:
                b = str2.split('/')[-1]
                b = re.split('[_ , .]', b)
                if a[1]==b[2] and a[2]==b[3] and a[3]==b[4]:
                    temp_target.append(str2)
                    #print(str1, str2)
        self.data_filenames_target1 = temp_target
        #print(self.data_filenames_target1)
        #print(self.data_filenames_input)

        self. transform = transform
        
        # DA-duplicate data
#        if self.transform: 
#            temp_input = self.data_filenames_input
#            temp_target1 = self.data_filenames_target1
#            temp_target2 = self.data_filenames_target2
#            for i in range(10):
#                self.data_filenames_input=self.data_filenames_input+temp_input
#                self.data_filenames_target1=self.data_filenames_target1 + temp_target1
#                self.data_filenames_target2=self.data_filenames_target2 + temp_target2

    def __getitem__(self, index):
#        print('Index',index)
#        print(len(self.data_filenames_input))
         input = load_data(self.data_filenames_input[index])
#         print(input.shape)
         input = input[5::,]
#        print('First Trans',input.size())
#        input = input[2,]
#        print('Second Trans',input.size())
#        input = input.view(1, input.size()[0], input.size()[1], input.size()[2], input.size()[3])
#        print('Third Trans',input.size())
         target = load_data2(self.data_filenames_target1[index])
#         print(target.shape)
         target = target[1,]
#        target1 = target1[2,]
#        target1 = target1.view(1, target1.size()[0], target1.size()[1], target1.size()[2])
   
#        target2 = target2[5,]
#        target2 = target2.view(1, target2.size()[0], target2.size()[1], target2.size()[2])
#        print(input.shape, target1.shape, target2.shape)
#        print(input.shape, target1.shape, target2.shape)
#
#        print(input.size()[0],input.size()[1],input.size()[2],input.size()[3])
        
        # DA-operation
#        if self.transform:
#            # DA-translate(random select)
#            indx = random.randint(0,160) #x
#            input = input[:, indx:indx+56, :, :]
#            target1 = target1[:, indx:indx+56, :, :]
#            target2 = target2[:, indx:indx+56, :, :]
#            indx = random.randint(0,64) #y
#            input = input[:, :, :, indx:indx+56]
#            target1 = target1[:, :, :, indx:indx+56]
#            target2 = target2[:, :, :, indx:indx+56]
#            #indx = random.randint(0,12) #z
            #input = input[:, :, indx:indx+12, :]
            #target1 = target1[:, :, indx:indx+12, :]
            #target2 = target2[:, :, indx:indx+12, :]
            # DA-flip
            #indx = random.randint(1,6) #x, y, z, xy, yz, xz
            #if indx == 1 or indx == 4 or indx == 6: #x
            #    input = torch.flip(input,[1])
            #    target1 = torch.flip(target1,[1])
            #    target2 = torch.flip(target2,[1])
            #if indx == 2 or indx == 4 or indx == 5: #y
            #    input = torch.flip(input,[3])
            #    target1 = torch.flip(target1,[3])
            #    target2 = torch.flip(target2,[3])
            #if indx == 3 or indx == 5 or indx == 6: #z
            #    input = torch.flip(input,[2])
            #    target1 = torch.flip(target1,[2])
            #    target2 = torch.flip(target2,[2])          
         return input, target

    def __len__(self):
        return len(self.data_filenames_input)
