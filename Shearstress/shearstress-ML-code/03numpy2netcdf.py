import numpy as np
from netCDF4 import Dataset
import os

def np2netcdf(name, out):
    Vel = np.load(name)
    fname = os.path.basename(name)[:-4]
    #Coord = np.load("./test/input/"+fname+".npy")
    Coord = np.load("./train/input/VelP_0_016000_2d_5.npy")
    print(Vel.shape)
    print(Coord.shape)
    outname = out+ "/" + fname + ".nc"
    print(outname)
    Vel_out = Dataset(outname, "w")
    
    MX = Vel_out.createDimension("MX", Vel.shape[2])
    #MY = Vel_out.createDimension("MY", Vel.shape[2])
    MZ = Vel_out.createDimension("MZ", Vel.shape[1])
    X = Vel_out.createVariable("X", "f4", ("MZ","MX"))
    Y = Vel_out.createVariable("Y", "f4", ("MZ","MX"))
    Z = Vel_out.createVariable("Z", "f4", ("MZ","MX"))
    #Vel_x = Vel_out.createVariable("Vel_x", "f4", ("MZ","MY","MX"))
    #Vel_y = Vel_out.createVariable("Vel_y", "f4", ("MZ","MY","MX"))
    Vel_z = Vel_out.createVariable("Vel_z", "f4", ("MZ","MX"))
    
    X[:,:] = Coord[3,]
    #Y[:,:] = Coord[4,]
    Z[:,:] = Coord[5,]
    #Vel_x[:,:] = Vel[0,]
    #Vel_y[:,:] = Vel[1,]
    Vel_z[:,:] = Vel[2,]
    
    Vel_out.close()
    
    
    


def sdmkdir(x):
    if not os.path.isdir(x):
        os.makedirs(x)

if __name__=='__main__':
    path = './test/output/'
    out = './test/output/netcdf'
    #path = './test/input/'
    #out = './'
    sdmkdir(out) 
    all =[]
    for x in os.listdir(path):
        if x.endswith(".npy"):
            all.append(os.path.join(path,x))
    print(all)
    for name in all:
        fname = os.path.basename(name)[:-4]
        print(fname)
        np2netcdf(name, out)
