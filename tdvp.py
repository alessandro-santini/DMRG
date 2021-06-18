import MPS_class as MPS
import MPO_class as MPO
import contraction_utilities as contract
import numpy as np
import numpy.linalg as LA
from ncon import ncon
import matplotlib.pyplot as plt
import dmrg1 
from LanczosRoutines import expm_krylov_lanczos

## tensor contraction for minimization
##         +--M--+
##         |  |  |
## H_eff = L--H--R
##         |  |  |
##         +--M--+       
def apply_Heff(L,H,R,M):
    return ncon([L,H,R,M],[[1,2,-1],[2,5,3,-2],[4,5,-3],[1,3,4]])
def apply_Hfree(L,R,C):
    return ncon([L,R,C],[[1,3,-1],[2,3,-2],[1,2]])

def build_Heff(L,H,R):
    return ncon([L,H,R],[[-1,1,-4],[1,2,-2,-5],[-3,2,-6]])
def build_Hfree(L,R):
    return ncon([L,R],[[-1,1,-3],[-2,1,-4]])

def local_exponentiation(method,M,L,H,R,delta,maxit=10):
    if method == 'H':
            Afunc = lambda x: apply_Heff(L, H, R, x.reshape(M.shape)).ravel()
    if method == 'Hfree':
            Afunc = lambda x: apply_Hfree(L, R, x.reshape(M.shape)).ravel()
    v = expm_krylov_lanczos(Afunc, M.ravel(), 1j*delta/2, maxit)
    return v/LA.norm(v)
    
class TDVP:
    def __init__(self,MPS_,H):
        self.MPS = MPS.MPS(MPS_.L,MPS_.chim,MPS_.d)
        self.MPS.M = MPS_.M.copy()
        self.MPO = H
        self.L = self.MPS.L
        
    def initialize(self):
        L = self.L

        self.RT = [0 for x in range(self.L+1)]
        self.LT = [0 for x in range(self.L+1)]
        
        self.RT[L]  = np.ones((1,1,1))
        self.LT[-1] = np.ones((1,1,1))
        
        # Generates R tensors
        for j in range(L-1,0,-1):
            self.RT[j] = contract.contract_right(self.MPS.M[j], self.MPO.W[j], self.MPS.M[j].conj(), self.RT[j+1])
       
    def right_sweep(self,delta):
        for i in range(self.L):
            M = self.MPS.M[i]
            shpM = M.shape
            psi = local_exponentiation('H',M, self.LT[i-1], self.MPO.W[i], self.RT[i+1],delta)
            
            M = psi.reshape(shpM[0]*shpM[1],shpM[2])
            
            U,S,V = LA.svd(M,full_matrices=False)
            S/=LA.norm(S)
            A = U.reshape(shpM[0],shpM[1], S.size)
            
            self.LT[i]  = contract.contract_left(A, self.MPO.W[i], A.conj(), self.LT[i-1])
            self.MPS.M[i] = A
                        
            if i != self.L-1:
                C = np.diag(S)@V
                psi   = local_exponentiation('Hfree',C, self.LT[i],' ', self.RT[i+1],-delta)
                C = psi.reshape(C.shape)
                self.MPS.M[i+1] = ncon([C, self.MPS.M[i+1]],[[-1,1],[1,-2,-3]])
            
    def left_sweep(self,delta):
        for i in range(self.L-1,-1,-1):
            M = self.MPS.M[i]
            shpM = M.shape
            
            psi = local_exponentiation('H',M, self.LT[i-1], self.MPO.W[i], self.RT[i+1],delta)
            M = psi.reshape(shpM[0],shpM[1]*shpM[2])
            
            U,S,V = LA.svd(M,full_matrices=False)
            S/=LA.norm(S)
            B = V.reshape(S.size,shpM[1],shpM[2])
            self.RT[i]  = contract.contract_right(B, self.MPO.W[i], B.conj(), self.RT[i+1])
            self.MPS.M[i] = B
            
            if i != 0:
                C = U@np.diag(S)
                psi   = local_exponentiation('Hfree',C, self.LT[i-1],' ', self.RT[i],-delta)
                C = psi.reshape(C.shape)
                self.MPS.M[i-1] = ncon([self.MPS.M[i-1],C],[[-1,-2,1],[1,-3]])
                
    def time_step(self,delta):
        self.right_sweep(delta)
        self.left_sweep(delta)
        
L = 64
h = 0.
delta = 1.5
chi = 64
h = 0.
H = MPO.XXZMPO(L, delta, h)
alg = dmrg1.DMRG1(H)

alg.initialize(chi)
for n in range(20):
    alg.right_sweep()
    alg.left_sweep()

print('en:', alg.MPO.contractMPOMPS(alg.MPS).real)
#%%
sigma_z = np.array([[1,0],[0,-1]])
deltaf = -2

# Hf = MPO.XXZMPO(L, deltaf, h)
Hf = MPO.IsingMPO(L, 0.5)
alg1 = TDVP(alg.MPS, Hf)
alg1.initialize()
dt = 0.1

Mz = MPO.getMzMPO(L)
mz = []
et = []

mz.append(Mz.contractMPOMPS(alg1.MPS))
et.append(Hf.contractMPOMPS(alg1.MPS).real)
Sent = []
t = []
mz0 = []
t.append(0.)
Sent.append(alg1.MPS.compute_EntEntropy())

for n in range(1,101):
    print(n)
    t.append((n)*dt)
    alg1.time_step(dt)
    mz.append(Mz.contractMPOMPS(alg1.MPS))
    et.append(Hf.contractMPOMPS(alg1.MPS).real)
    Sent.append(alg1.MPS.compute_EntEntropy())
    mz0.append( ncon([alg1.MPS.M[0],alg1.MPS.M[0].conj(),sigma_z],[[1,2,3],[1,4,3],[2,4]]).real)
#%%
import seaborn as sns
from scipy.optimize import curve_fit

def lin(x,a,b):
    return a*x+b
t    = np.array(t)
Sent = np.array(Sent)
plt.rc('text',usetex=True)
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.imshow(Sent,origin='lower',aspect='auto',extent=[1,L,0,t[-1]],cmap = sns.color_palette('hot',as_cmap=True))
plt.xlabel("$L$", fontsize=22)
plt.ylabel("$t$", fontsize=22)
plt.tick_params(labelsize=20)
plt.subplot(1,2,2)
for x in range(0,64,8):
    plt.plot(t,Sent[:, x],label=f'$L={x+1}$')
plt.ylabel("$S_L$", fontsize=22)
plt.xlabel("$t$", fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(labelsize=20)

plt.tight_layout()


popt, pcov = curve_fit(lin, t[(t<5)*(t>2)], Sent[(t<5)*(t>2),L//2])
x = np.linspace(1,8.5,3)
plt.plot(x,lin(x,popt[0],popt[1])-2.5,'--k')