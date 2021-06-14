import numpy as np
import numpy.linalg as LA
from ncon import ncon
import contraction_utilities as contract
from scipy.linalg import expm


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
    
    # M-M-M-M-M
    # | | | | |
    # W-W-W-W-W
    # | | | | |
    # M-M-M-M-M    
    def contractMPOMPS(self, MPS):
       if(MPS.L != self.L): raise Exception('MPS MPO length are different')
       Rtemp = np.ones((1,1,1),dtype=np.complex128)
       for i in range(self.L-1,0,-1):
           Rtemp = contract.contract_right(MPS.M[i], self.W[i], MPS.M[i].conj(), Rtemp)
       return contract.contract_right(MPS.M[0], self.W[0], MPS.M[0].conj(), Rtemp)[0][0][0]
        
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

def getMzMPO(L):
    MzMPO = MPO(L,2)
    mz  = np.zeros((2,2,2,2)); mzl = np.zeros((1,2,2,2)); mzr = np.zeros((2,1,2,2))
    sigma_z  = np.array([[1, 0], [0,-1]]);  Id  = np.array([[1, 0], [0, 1]])
    # Bulk
    mz[0,0,:,:] = Id; mz[1,0,:,:] = sigma_z
    mz[1,1,:,:] = Id
    # Boundary
    mzr[0,0,:,:] = Id; mzr[1,0,:,:] = sigma_z
    mzl[0,0,:,:] = sigma_z; mzl[0,1,:,:] = Id    
    # Set the MPO
    MzMPO.W[0] = mzl; MzMPO.W[L-1] = mzr
    for i in range(1,L-1):
        MzMPO.W[i] = mz
    return MzMPO
    
def getMxMPO(L):
    MxMPO = MPO(L,2)
    mx  = np.zeros((2,2,2,2)); mxl = np.zeros((1,2,2,2)); mxr = np.zeros((2,1,2,2))
    sigma_x  = np.array([[0, 1], [1, 0]]);  Id  = np.array([[1, 0], [0, 1]])
    # Bulk
    mx[0,0,:,:] = Id; mx[1,0,:,:] = sigma_x
    mx[1,1,:,:] = Id
    # Boundary
    mxr[0,0,:,:] = Id; mxr[1,0,:,:] = sigma_x
    mxl[0,0,:,:] = sigma_x; mxl[0,1,:,:] = Id    
    # Set the MPO
    MxMPO.W[0] = mxl; MxMPO.W[L-1] = mxr
    for i in range(1,L-1):
        MxMPO.W[i] = mx
    return MxMPO

def getStagMzMPO(L):
    sMzMPO = MPO(L,2)
    mzl = np.zeros((1,2,2,2)); mzr = np.zeros((2,1,2,2))
    sigma_z  = np.array([[1, 0], [0,-1]]);  Id  = np.array([[1, 0], [0, 1]])
    # Boundary
    mzr[0,0,:,:] = Id; 
    mzr[1,0,:,:] = sigma_z*(-1)**(L-1)
    mzl[0,0,:,:] = sigma_z; mzl[0,1,:,:] = Id    
    # Set the MPO
    sMzMPO.W[0] = mzl; 
    sMzMPO.W[L-1] = mzr
    for i in range(1,L-1):
        # Bulk
        mz  = np.zeros((2,2,2,2));
        mz[0,0,:,:] = Id; 
        mz[1,0,:,:] = sigma_z*(-1)**i
        mz[1,1,:,:] = Id
        sMzMPO.W[i] = mz
    return sMzMPO

def ComputeCorrFunction(MPS,i,j,Opi,Opj):   
    shpMi = MPS.M[i].shape
    shpMj = MPS.M[j].shape
    L = np.zeros((shpMi[0],1,shpMi[0]))
    Rtemp = np.zeros((shpMj[2],1,shpMj[2]))
    
    L[:,0,:] = np.eye(shpMi[0])
    Rtemp[:,0,:] = np.eye(shpMj[2])
    
    Rtemp = contract.contract_right(MPS.M[j],Opj,MPS.M[j].conj(),Rtemp)
    for k in range(j-1, i, -1):
           Rtemp = contract.contract_right(MPS.M[k], np.eye(MPS.d).reshape(1,1,2,2), MPS.M[k].conj(), Rtemp)
    R = contract.contract_right(MPS.M[i], Opi, MPS.M[i].conj(), Rtemp)
    return ncon([L,R],[[1,2,3],[1,2,3]])


        