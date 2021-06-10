import numpy as np
import numpy.linalg as LA
from ncon import ncon

class MPS:
    def __init__(self,L,chim,d):
        # L: length of the tensor train
        # chim: maximum bond dimension
        # d: local Hilbert Space dimension
        
        #   Index order
        #   0--M--2
        #      |
        #      1
        self.L = L
        self.chim = chim
        self.d = d
        self.M = [0 for x in range(L)]
        self.Sab = np.zeros(self.L-1)
        
    def initializeMPS(self, chi):
        """Initialize a random MPS with bond dimension chi
        local hilbert space dim d and length L"""
        d = self.d
        L = self.L
        self.M[0]   = np.random.rand(1,   d, chi)
        self.M[L-1] = np.random.rand(chi, d, 1)
        for i in range(1,L-1):
            self.M[i] = np.random.rand(chi,d,chi)   
    
    def right_normalize(self):
        for i in range(self.L-1,0,-1):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0], shpM[1]*shpM[2]), full_matrices=False)
            self.M[i] = V.reshape(S.size, shpM[1], shpM[2])
            self.M[i-1] = ncon([self.M[i-1],U*S],[[-1,-2,1],[1,-3]])
    
    def left_normalize(self):
        for i in range(0, self.L-1):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0]*shpM[1], shpM[2]), full_matrices=False)
            self.M[i] =  U.reshape(shpM[0], shpM[1], S.size)
            self.M[i+1] = ncon([np.diag(S)@V, self.M[i+1]],[[-1,1],[1,-2,-3]])
    
    def check_normalization(self, which='R'):
        if which == 'R':
            for i in range(self.L):
                X = [self.M[i][:,j,:]@self.M[i][:,j,:].T for j in range(self.d)]
                print('site',i,np.allclose(sum(X),np.eye(self.M[i].shape[0]))) 
        if which == 'L':
            for i in range(self.L):
                X = [self.M[i][:,j,:].T@self.M[i][:,j,:] for j in range(self.d)]
                print('site',i,np.allclose(sum(X),np.eye(self.M[i].shape[2])))