import torch
import torch.nn as nn
import torch.nn.init as init

torch.set_printoptions(precision=8)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential( 
                  nn.Conv2d(8, 8, kernel_size = (7,3), stride = (1,1), padding = (3,1),bias=False),
#                  nn.BatchNorm2d(8),
                  nn.LeakyReLU(0.1) )
#                  nn.BatchNorm2(8),

        self.layer2 = nn.Sequential(
                  nn.Conv2d(8, 16, kernel_size = (9,3), stride = (5,2), padding = (3,1),bias=False),
#                  nn.BatchNorm2d(16),
                  nn.LeakyReLU(0.1) )
#                  nn.BatchNorm2(16),

        self.layer3 = nn.Sequential(
                  nn.Conv2d(16, 32, kernel_size = (7,3), stride = (4,2), padding = (3,1),bias=False),
#                  nn.BatchNorm2d(32),
                  nn.LeakyReLU(0.1) )
#                 nn.BatchNorm2(32),

        self.layer4 = nn.Sequential(
                  nn.Conv2d(32, 64, kernel_size = (7,3), stride = (4,2), padding = (3,1),bias=False),
#                  nn.BatchNorm2d(64),
                  nn.LeakyReLU(0.1) )
#                 nn.BatchNorm2(64),

        self.layer5 = nn.Sequential(
                  nn.ConvTranspose2d(64, 32, kernel_size = (6,4), stride = (4,2), padding = (1,1),bias=False),
#                  nn.BatchNorm2d(32),
                  nn.PReLU() )
#                 nn.BatchNorm3d(32),

        self.layer6 = nn.Sequential(
                  nn.ConvTranspose2d(32, 16, kernel_size = (6,4), stride = (4,2), padding = (1,1),bias=False),
#                  nn.BatchNorm2d(16),
                  nn.PReLU() )
#                  nn.BatchNorm3d(16),

        self.layer7 = nn.Sequential(
                  nn.ConvTranspose2d(16, 8, kernel_size = (9,4), stride = (5,2), padding = (2,1),bias=False),
#                  nn.BatchNorm2d(8),
                  nn.PReLU() )
#                  nn.BatchNorm3d(8),

        self.output = nn.Sequential(
                  nn.Conv2d(8, 1, kernel_size = (7,3), stride = (1,1), padding = (3,1)),
#                  nn.BatchNorm2d(1),
                  nn.PReLU() )
        

    def forward(self, x):
#        x1 = x
        #print(x.size())
        #x = torch.mean(x,1)
        #x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        #print(x.size())
        #x[:, 2, :, :, :] = x[:, 2, :, :, :]/0.4+0.5 # normalization of v and w
#        print(torch.mean(x,[0,2,3]))
#        print(torch.var(x,[0,2,3],unbiased=False))
        x = self.layer1(x)
#        print(self.layer1[1].running_mean)
#        print(self.layer1[1].running_var)
       #print(torch.mean(x,[0,2,3]))
       #print(torch.var(x,[0,2,3],unbiased=False))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.output(x)
        #x[:, 2, :, :, :] = x[:, 2, :, :, :]*0.4-0.2
        #print(x.size())
        avg = x

#        x = x.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4],1)
#        x = (x1-x)**2
#        x = torch.mean(x,5)
#
#        x[:, :, :, :, :] = x[:, :, :, :, :]*40 #normalization of uu
#        x = self.layer2_1(x)
#        x = self.layer2_2(x)
#        x = self.layer2_3(x)
#        x = self.layer2_4(x)
#        x = self.layer2_5(x)
#        x = self.layer2_6(x)
#        x = self.layer2_7(x)
#        x = self.output2(x)
#        #x[:, 0:2, :, :, :] = x[:, 0:2, :, :, :]/120
#        x[:, :, :, :, :] = x[:, :, :, :, :]/40
#        uu = x
        #print(uu.size())
#        return avg, uu
        return avg

#    def _initialize_weights(self):

#        init.orthogonal_(self.layer1.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer2.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer3.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer4.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer5.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer6.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer7.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.output.weight)
#
#        init.orthogonal_(self.layer2_1.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer2_2.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer2_3.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer2_4.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer2_5.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer2_6.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.layer2_7.weight, init.calculate_gain('leaky_relu', 0.1))
#        init.orthogonal_(self.output2.weight)


#    def _initialize_weights(m):
#        classname = m.__class__.__name__
#        if classname.find('Conv') != -1:
#            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#        elif classname.find('BatchNorm2d') != -1:
#            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#            torch.nn.init.constant_(m.bias.data, 0.0)
