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
        for i in range(self.MPS.L):
            self.MPS.M[i] = MPS_.M[i].copy()
            self.MPS.Svr[i] = MPS_.Svr[i].copy()
        self.MPS.Svr[0] = np.array([1])
        self.L = self.MPS.L
        
    def set_UXXZ(self,h,delta,dt):
        d   = self.MPS.d
        Sp  = np.array([[0,1],[0,0]])
        Sm  = np.array([[0,0],[1,0]])
        Sz  = 0.5*np.array([[1, 0], [0,-1]])
        
        hi  = -h*Sz 
        hij = .5*np.kron(Sm,Sp)+.5*np.kron(Sp,Sm)+delta*np.kron(Sz,Sz)   
        U1  = expm(-1j*dt*hi)
        U2  = expm(-1j*dt*hij).reshape(d,d,d,d)
        self.U1 = U1
        self.U2 = U2
        
    def evolve(self,chi):
        # One-body term:
        L = self.L
        d = self.MPS.d
        for i in range(L):
            self.MPS.M[i] = ncon([self.MPS.M[i],self.U1],[[-1,1,-3],[1,-2]])
        # Two-body interaction:
        # Even-Bonds
        for i in np.concatenate((np.arange(0,L,2),np.arange(1,L-1,2))):
            
            Mi   = ncon([np.diag(self.MPS.Svr[i+1]),self.MPS.M[i]],[[-1,1],[1,-2,-3]])
            Mj   = self.MPS.M[i+1]
            
            shpMi = Mi.shape
            shpMj = Mj.shape
            theta = ncon([Mi,Mj,self.U2],[[-1,1,2],[2,3,-3],[1,3,-2,-4]])
            theta = theta.reshape(shpMi[0]*d,shpMj[2]*d)
            U,S,V = LA.svd(theta,full_matrices=False)
            if S.size > chi:
                U = U[:,:chi]
                S = S[:chi]
                V = V[:chi,:]
            S /= LA.norm(S)
            self.MPS.M[i+1] = V.reshape(S.size,d,shpMj[2])
            A = U.reshape(shpMi[0],d,S.size)
            self.MPS.M[i]   = ncon([LA.inv(np.diag(self.MPS.Svr[i+1])),A,np.diag(S)],[[-1,1],[1,-2,2],[2,-3]])
            self.MPS.Svr[i+2] = S
            
L = 64
h = 0.
delta = -1.1
chi = 64

H = MPO.XXZMPO(L, delta, h)
alg = dmrg1.DMRG1(H)

alg.initialize(chi)
for n in range(10):
    alg.right_sweep()
    alg.left_sweep()
            
deltaf = 0.9
T = 5
dt = 0.01
nsteps = int(T/dt)
alg1 = TEBD(alg.MPS)
alg1.set_UXXZ(h, delta, dt)

Mz = MPO.getMzMPO(L)
Hf = MPO.XXZMPO(L, deltaf, h)
mz = np.zeros(nsteps+1)
en = np.zeros(nsteps+1)
t  = np.zeros(nsteps+1)

mz[0] = Mz.contractMPOMPS(alg1.MPS).real 
en[0] = Hf.contractMPOMPS(alg1.MPS).real
t[0]  = 0.
Sent = np.zeros((L, nsteps+1))

Svls = alg1.MPS.Svr[1:]
Sent[:,0] = alg1.MPS.compute_EntEntropy()

for n in range(nsteps):
    alg1.evolve(64)
    mz[n+1] = Mz.contractMPOMPS(alg1.MPS)
    en[n+1] = Hf.contractMPOMPS(alg1.MPS)
    t[n+1]  = (n+1)*dt
    print('t',t[n+1],'Delta en',en[n+1]-en[0])
    Svls = alg1.MPS.Svr[1:]  
    Sent[:,n+1] = alg1.MPS.compute_EntEntropy()
    
#%% ent entropy plot
import matplotlib
plt.rc('text',usetex=True)
plt.imshow(Sent.T,aspect='auto',origin='lower',extent=[0,L,0,T],cmap=plt.cm.hot,norm=matplotlib.colors.LogNorm(vmin=1e-5,vmax=Sent.max()))
plt.xlabel("$l$", fontsize=32)
plt.ylabel("$t$", fontsize=32)
plt.tick_params(labelsize=20)
plt.xticks(np.arange(0,61,10))
plt.tight_layout()
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
