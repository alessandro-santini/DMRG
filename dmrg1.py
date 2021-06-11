import MPS_class as MPS
import MPO_class as MPO
import contraction_utilities as contract

import numpy as np
import numpy.linalg as LA
from ncon import ncon
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

## tensor contraction for minimization
##         +-   -+
##         |  |  |
## H_eff = L--H--R
##         |  |  |
##         +-- --+
def build_Heff(L,H,R):
    return ncon([L,H,R],[[-1,1,-4],[1,2,-2,-5],[-3,2,-6]])

## tensor contraction for minimization
##         +--M--+
##         |  |  |
## H_eff = L--H--R
##         |  |  |
##         +--M--+       
def apply_Heff(L,H,R,M):
    return ncon([L,H,R,M.conj()],[[-1,2,1],[2,5,-2,3],[-3,5,4],[1,3,4]])

def local_minimization(M,L,H,R,Lsteps=10):
        psi = M.ravel()
        
        if LA.norm(psi) < 1e-16: psi = np.random.rand(psi.size)
        psi /= LA.norm(psi)
        
        V = np.zeros([len(psi), Lsteps], dtype=np.float64)
        alpha = np.zeros(Lsteps,dtype=np.float64)
        beta  = np.zeros(Lsteps-1,dtype=np.float64)
        V[:, 0]  = psi
        V[:, 1]  = apply_Heff(L,H,R,M).ravel()
        alpha[0] = np.vdot(V[:,1], V[:,0])
        V[:, 1]  -= alpha[0]*V[:, 0]
        for j in range(1,Lsteps-1):
            beta[j-1] = LA.norm(V[:, j])
            V[:, j]  /= max(beta[j-1],1e-16)
            V[:, j+1] = apply_Heff(L,H,R,V[:,j].reshape(M.shape)).ravel()
            alpha[j]  = np.vdot(V[:,j+1],V[:,j])
            V[:,j+1] -= alpha[j]*V[:,j] + beta[j-1]*V[:,j-1]
        eig, w = eigh_tridiagonal(alpha,beta)    
        psi = V@w[:,0]
        return psi, eig[0]

class DMRG1:
    def __init__(self, H, chim):
        self.MPS = MPS.MPS(H.L, chim, H.d)
        self.MPO  = H
        # check H.L == MPS.L
        self.L = self.MPS.L
        self.E = 0.
        
    def initialize(self,chi):
        # Generate a randomMPS and put it in right
        # canonical form
        self.MPS.initializeMPS(chi)      
        self.MPS.right_normalize()
        L = self.L

        self.RT = [0 for x in range(self.L+1)]
        self.LT = [0 for x in range(self.L+1)]
        
        self.RT[L]  = np.ones((1,1,1))
        self.LT[-1] = np.ones((1,1,1))
        
        # Generates R tensors
        for j in range(L-1,0,-1):
            self.RT[j] = contract.contract_right(self.MPS.M[j], self.MPO.W[j], self.MPS.M[j].conj(), self.RT[j+1])

    def check_convergence(self):
        self.H2 = MPO.MPO(self.MPO.L,self.MPO.d)
        for x in range(self.L):
            shpW = self.MPO.W[x].shape
            self.H2.W[x] = ncon([self.MPO.W[x],self.MPO.W[x]],[[-1,-3,1,-6],[-2,-4,-5,1]])
            self.H2.W[x] = self.H2.W[x].reshape(shpW[0]*shpW[0],shpW[1]*shpW[1],shpW[2],shpW[3])
        E2 = self.H2.contractMPOMPS(self.MPS)
        E  = self.MPO.contractMPOMPS(self.MPS)
        return (E2 - E**2)/self.L

    def right_sweep(self):
        for i in range(self.L):
            M = self.MPS.M[i]
            shpM = M.shape
            psi, e = local_minimization(M, self.LT[i-1], self.MPO.W[i], self.RT[i+1])
            
            U,S,V = LA.svd(psi.reshape(shpM[0]*shpM[1],shpM[2]),full_matrices=False)
            
            A = U.reshape(shpM[0],shpM[1], S.size)
            self.LT[i]  = contract.contract_left(A, self.MPO.W[i], A.conj(), self.LT[i-1])
            self.MPS.M[i] = A
            
            if i != self.L-1:
                SV = (np.diag(S)@V)
                self.MPS.M[i+1] = ncon([SV, self.MPS.M[i+1]],[[-1,1],[1,-2,-3]])
            self.E = e            
            
    def left_sweep(self):
        for i in range(self.L-1,-1,-1):
            M = self.MPS.M[i]
            shpM = M.shape
            psi, e = local_minimization(M, self.LT[i-1], self.MPO.W[i], self.RT[i+1])
            U,S,V = LA.svd(psi.reshape(shpM[0],shpM[1]*shpM[2]),full_matrices=False)
            
            B = V.reshape(S.size,shpM[1],shpM[2])
            self.RT[i]  = contract.contract_right(B, self.MPO.W[i], B.conj(), self.RT[i+1])
            self.MPS.M[i] = B
            
            if i != 0:
                US = U@np.diag(S)
                self.MPS.M[i-1] = ncon([self.MPS.M[i-1],US],[[-1,-2,1],[1,-3]])
            self.E = e
        return    
        
L = 128
delta_space = np.linspace(-0.9,2.,4)
chim, chi = 170, 200 

Mz  = MPO.getMzMPO(L)
Mx  = MPO.getMxMPO(L)
SMz = MPO.getStagMzMPO(L)
        
mz  = np.zeros_like(delta_space)
mx  = np.zeros_like(delta_space)
smz = np.zeros_like(delta_space)

H = MPO.XXZMPO(L, -1, 0.)
alg = DMRG1(H, chim)
alg.initialize(chi)

for i in range(40):
    alg.right_sweep()
    alg.left_sweep()
    
sigma_z = np.array([[1,0],[0,-1]])
sigma_z = sigma_z.reshape(1,1,2,2)
Cij = np.zeros(64)
alg.MPS.mix_normalize(i)

for k,j in enumerate(range(33,97)):
    Cij[k] = MPO.ComputeCorrFunction(alg.MPS, 32,j,sigma_z,sigma_z )

#%%
for jd,delta in enumerate(delta_space):
    H = MPO.XXZMPO(L, delta, 0.)
    alg = DMRG1(H, chim)
    alg.initialize(chi)
    for i in range(4):
        alg.right_sweep()
        alg.left_sweep()
    mz[jd] = Mz.contractMPOMPS(alg.MPS).real/alg.L
    mx[jd] = Mx.contractMPOMPS(alg.MPS).real/alg.L
    smz[jd] = SMz.contractMPOMPS(alg.MPS).real/alg.L 
    #print(alg.check_convergence())
#%%
plt.plot(delta_space,np.abs(mz),label=r'$m_z$')
#plt.plot(delta_space,mx,label=r'$m_x$')
plt.plot(delta_space,np.abs(smz),label=r'$m_z^\mathrm{stag}$')

plt.tick_params(axis='both',which='major',labelsize=16)

plt.xlabel("$h$",fontsize=22)
plt.legend(fontsize=20)

plt.tight_layout()