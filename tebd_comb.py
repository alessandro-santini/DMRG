import MPS_class as MPS
import MPO_class as MPO
import contraction_utilities as contract

import numpy as np
import numpy.linalg as LA
from ncon import ncon
from scipy.linalg import expm
import matplotlib.pyplot as plt

import dmrg1 

class TEBD():
    def __init__(self, MPS_):
        self.MPS = MPS.MPS(MPS_.L,MPS_.chim,MPS_.d)
        self.MPS.M = MPS_.M.copy()
        self.L = self.MPS.L
    def allup(self):
        for x in range(self.L):
            self.MPS.M[x] = np.zeros((1,2,1))
            self.MPS.M[x][0,0,0] = 1.
    def set_UXXZ(self,h,delta,dt):
        d   = self.MPS.d
        L   = self.L
        
        Sp  = np.array([[0,1],[0,0]])
        Sm  = np.array([[0,0],[1,0]])
        Sz  = 0.5*np.array([[1, 0], [0,-1]])
        Id  = np.eye(2)
        
        hi  = -h*Sz
        hij = .5*np.kron(Sm,Sp)+.5*np.kron(Sp,Sm)+delta*np.kron(Sz,Sz)   
        U1  = [expm(-1j*dt*hi).reshape(1,1,d,d) for x in range(L)]
        
        U2  = expm(-1j*dt*hij)
        #U,S,V = LA.svd(U2,full_matrices=False)
        Q,R = LA.qr(U2)
        #U2L = U@np.diag(np.sqrt(S))
        #U2R = np.diag(np.sqrt(S))@V
        U2L = Q
        U2R = R
        S = np.diag(Q)
        self.U = [0 for x in range(L)]
        
        Uo = [0 for x in range(L)]
        Ue = [0 for x in range(L)]
        
        Uo[0]  = Id.reshape(1,1,d,d)
        Uo[-1] = Id.reshape(1,1,d,d)
        
        for x in range(0,L,2):
            Ue[x]   = U2L.reshape(1,S.size,d,d)
            Ue[x+1] = U2R.reshape(S.size,1,d,d)
        for x in range(1,L-1,2):
            Uo[x]   = U2L.reshape(1,S.size,d,d)
            Uo[x+1] = U2R.reshape(S.size,1,d,d)
        self.U1 = U1
        self.Ue = Ue
        self.Uo = Uo
        #for x in range(L):
        #    shpe = Ue[x].shape
        #    shpo = Uo[x].shape
        #    shp1 = U1[x].shape
        #    self.U[x] = ncon([Ue[x],Uo[x],U1[x]],[[-1,-4,1,-8],[-2,-5,2,1],[-3,-6,-7,2]])
        #    self.U[x] = self.U[x].reshape(shpe[0]*shpo[0]*shp1[0],shpe[1]*shpo[1]*shp1[1],shp1[2],shpe[3])
        
    def evolve(self,chi):
        for x in range(self.L):
            for U in [self.Ue[x],self.Uo[x],self.U1[x]]:
                M = self.MPS.M[x]
                #U = self.U[x]
                
                shpM = M.shape
                shpU = U.shape
                
                M  = ncon([M,U],[[-1,1,-4],[-2,-5,1,-3]])
                M  = M.reshape(shpM[0]*shpU[0],shpU[3],shpM[2]*shpU[1])
                self.MPS.M[x] = M
        #self.MPS.right_normalize_and_truncate(chi)
#%%        
L = 64
h = 0.
delta = -1.
chi = 64
h = 0.
H = MPO.XXZMPO(L, delta, h)
alg = dmrg1.DMRG1(H)

alg.initialize(chi)
for n in range(10):
    alg.right_sweep()
    alg.left_sweep()
print('en:',alg.MPO.contractMPOMPS(alg.MPS))

#%%
alg1 = TEBD(alg.MPS)
deltaf = -2
h  = 0.
dt = 0.01
alg1.set_UXXZ(h, deltaf, dt)
alg1.allup()
Hf = MPO.XXZMPO(L, deltaf, h)

Mz = MPO.getMzMPO(L)
Mx = MPO.getMxMPO(L)
mz = []
Sent = []
t = []

mz.append(Mz.contractMPOMPS(alg1.MPS).real/L)
Sent.append(alg1.MPS.compute_EntEntropy())
t.append(0.)

en0 = Hf.contractMPOMPS(alg1.MPS).real
print('en0',en0)
for n in range(10):
    alg1.evolve(chi)
    mz.append(Mz.contractMPOMPS(alg1.MPS).real/L)
    Sent.append(alg1.MPS.compute_EntEntropy())
    t.append(t[n]+dt)
    print('t',t[n+1],'errE',Hf.contractMPOMPS(alg1.MPS).real)