import numpy as np
import numpy.linalg as LA
from ncon import ncon

class MPO:
    def __init__(self,L,d):
        #   Index order
        #       2
        #       |
        #    0--W--1
        #       |
        #       3
        #
        # L: length of the tensor train
        # d: local Hilbert Space dimension
        self.L = L
        self.d = d
        self.W = [0 for x in range(L)]
        
def IsingMPO(L,h):
    Ham = MPO(L,2)
        
    # Pauli Matrices
    Sx = 0.5*np.array([[0, 1], [1, 0]])
    Sz = 0.5*np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the local bulk MPO
    H = np.zeros([3,3,2,2])
    H[0,0,:,:] = Id; H[2,2,:,:] = Id; H[2,0,:,:] = -h*Sz
    H[1,0,:,:] = Sx; H[2,1,:,:]= -Sx

    # Building the boundary MPOs
    HL = np.zeros((1,3,2,2))
    HL[0,:,:,:] = H[2,:,:,:]
    HR = np.zeros((3,1,2,2))
    HR[:,0,:,:] = H[:,0,:,:]

    Ham.W[0] = HL
    Ham.W[L-1] = HR
    for i in range(1,L-1):
        Ham.W[i] = H
    return Ham

def XXZMPO(L,delta, h):
    Ham = MPO(L,2)
    # Pauli Matrices
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    Sz = 0.5*np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the local bulk MPO
    H = np.zeros([5,5,2,2])
    H[0,0,:,:] =       Id
    H[1,0,:,:] =       Sp
    H[2,0,:,:] =       Sm
    H[3,0,:,:] =       Sz
    H[4,0,:,:] =    -h*Sz
    H[4,1,:,:] =    .5*Sm
    H[4,2,:,:] =    .5*Sp
    H[4,3,:,:] = delta*Sz
    H[4,4,:,:] =       Id
    
    # Building the boundary MPOs
    HL = np.zeros((1,5,2,2))
    HL[0,:,:,:] = H[4,:,:,:]
    HR = np.zeros((5,1,2,2))
    HR[:,0,:,:] = H[:,0,:,:]
    
    Ham.W[0] = HL
    Ham.W[L-1] = HR
    for i in range(1,L-1):
        Ham.W[i] = H
    return Ham
