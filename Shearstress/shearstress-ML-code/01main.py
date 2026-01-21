from __future__ import print_function
import argparse
from math import log10

import sys
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ConvNet
#from data import get_dataset
from dataset import DatasetFromFolder
#from SelfDefLoss import PhysicsLoss

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
#print(torch.cuda.device_count())
#print(torch.cuda.is_available())
#device = torch.device("cuda") #cuda or cpu
#torch.backends.cudnn.benchmark = True
device = torch.device("cpu") #cuda or cpu

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--seed', nargs='?', const=500, type=int, default=500, help='seed value to use')
parser.add_argument('--epochs', nargs='?', const=1000, type=int, default=1000, help='number of epochs')
parser.add_argument('--lr', nargs='?', const=0.001, type=float, default=0.001, help='learning rate')
parser.add_argument('--restart', nargs='?', const=0, type=int, default=0, help='restart switch and location')
opt = parser.parse_args()

torch.manual_seed(opt.seed)
nEpochs=opt.epochs
lr=opt.lr
restart=opt.restart

print('===> Loading datasets')
train_set = DatasetFromFolder('/gpfs/scratch/zexizhang/Sediment/data/train/input/','/gpfs/scratch/zexizhang/Sediment/data/train/target/', transform = False)
training_data_loader = DataLoader(dataset=train_set, batch_size=256, shuffle=True, num_workers=4)
test_set = DatasetFromFolder('/gpfs/scratch/zexizhang/Sediment/data/test/input/','/gpfs/scratch/zexizhang/Sediment/data/test/target/')
test_data_loader = DataLoader(dataset=test_set, batch_size=256, shuffle=True,num_workers=4)

print('===> Building model')
model = ConvNet()
print("Let's use", torch.cuda.device_count(), "GPUs!")
#model = nn.DataParallel(model)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr, weight_decay = 0.01)
scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)

if restart:
    checkpoint = torch.load("./trained_model/model_epoch_{}.tar".format(restart))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train(epoch):
    epoch_loss = 0
    epoch_mse = 0
    epoch_mdiv = 0
    criterion = nn.MSELoss()
    start = datetime.now()
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
#        print(input.size())
#        print(target.size())
        input = input.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
#        print(input.size())
#        print(target.size())
        prediction = model(input)        
#        print(prediction.size())
        loss= criterion(prediction, target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch, iteration, len(training_data_loader), loss.item()))
        print("Training MSE {}".format(criterion(prediction, target).item()))

    scheduler_lr.step()
    #print(scheduler_lr.get_lr())
    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, epoch_loss / len(training_data_loader)))
    with open('./trained_model/TrainingError.dat','a') as f:
        print("{:.8f}".format(epoch_loss / len(training_data_loader)), file=f)
    print("Epoch "+str(epoch)+" completed in: " + str(datetime.now() - start))

def test():
    avg_psnr = 0
    avg_pmse = 0
    avg_pmdiv = 0
    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            input, target  = batch[0].to(device), batch[1].to(device)
            #print(target1.shape, target2.shape)
            input = input.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
            target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
            prediction = model(input)
            criterion = nn.MSELoss()
            loss= criterion(prediction, target)#+criterion(prediction[1][:,]*40, target2[:,]*40)
            print("Test MSE {}".format(criterion(prediction, target).item()))
            #print(torch.mean(torch.mean((prediction-target)**2)))
            #print(torch.mean(torch.mean(torch.abs(prediction-target))))
            #print(target[0,0,200,:])
            #print(input[0,0,200,:])
            #print(prediction[0,0,200,:])
            #print(model.state_dict())
            psnr = 10 * log10(1 / (loss.item()))
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))
    with open('./trained_model/TestingError.dat', 'a') as f:
        print("{:.8f}".format(avg_psnr / len(test_data_loader)), file=f)

def checkpoint(epoch):
    model_out_path = "./trained_model/model_epoch_{}.tar".format(epoch)
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if restart:
    start = restart + 1
else:
    start = 1
    file = open('./trained_model/TrainingError.dat','w')
    file.close()
    file = open('./trained_model/TestingError.dat', 'w')
    file.close()


for epoch in range(start, nEpochs + start):
    train(epoch)
    if epoch % 100 == 0:
        checkpoint(epoch)
    test()
