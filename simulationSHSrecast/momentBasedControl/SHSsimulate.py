import numpy as np
import numpy.random as rnd
import poly as pl

        

class singleModeSHS:
    def __init__(self,x,u=None,f=None,g=None,Lam=[],Phi=[]):
        self.x = x
        self.u = u
        self.f = f
        self.g = g
        
        if g is not None:
            self.nW = g.shape[1]
        else:
            self.nW = 0
        
        self.Lam = Lam
        self.Phi = Phi
        
        if u is None:
            z = x
        else:
            z = (x,u)
            
        self.f_fun = pl.functify(f,z)
        self.g_fun = pl.functify(g,z)
        self.Lam_fun = [pl.functify(lam,z) for lam in Lam]
        self.Phi_fun = [pl.functify(phi,z) for phi in Phi]
            
    def step(self,dt,x,u=None):
        if u is None:
            z = (x,)
        else:
            z = (x,u)
            
        dx = np.zeros(len(x))
        if self.f_fun is not None:
            dx += self.f_fun(*z) * dt
            
        if self.g_fun is not None:
            dw = np.sqrt(dt) * rnd.randn(self.nW)
            dx += np.dot(self.g_fun(*z),dw)
            
        # Check if there are any jump processes
        if len(self.Lam_fun) == 0:
            # No jump processes
            return x + dx
        
        
        # Check if there is a jump
        lamVec = np.array([lam(*z) for lam in self.Lam_fun])
        lamVal = np.sum(lamVec)
        gam = rnd.rand()

        if gam > lamVal * dt:
            # No jump on this interval
            # Note: if lamVal < 0, it will never jump
            return x + dx

        # If there is a jump, figure out what it should be
        gam = rnd.rand()
        pVec = lamVec / lamVal
        pSum = np.cumsum(pVec)

        jumpInd = np.argwhere(gam < pSum)[0,0]

        dx += self.Phi_fun[jumpInd](*z) - x

        return x + dx


    def simulate(self,Time,x0,U=None):
        NumSteps = len(Time)
        
        X = np.zeros((NumSteps,len(x0)))
        X[0] = x0

        x = np.array(x0,copy=True)
        for k in range(NumSteps-1):
            if U is None:
                z = (x,)
            else:
                z = (x,U[k])

            dt = Time[k+1]-Time[k]

            x = self.step(dt,*z)
            X[k+1] = x

        return X