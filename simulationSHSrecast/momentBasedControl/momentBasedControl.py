import numpy as np
import numpy.random as rnd
import utils as ut
import scipy.linalg as la
import cvxpy as cvx
import scipy.special as sfun
import randSDP as rs
import poly as pl

        

#### Below is stuff that is specifically for the optimal control problem

def generator(x,h,f=None,g=None,Lam=[],Phi=[]):
    Lh = 0
    
    if (f is not None) or (g is not None):
        Jac = pl.jacobian(h,x)
        
    if f is not None:
        Lh = Lh + np.dot(Jac,f)
    
    if g is not None:
        Hes = pl.jacobian(Jac,x)
        Lh = Lh + .5 * np.trace(ut.dot(Hes,g,g.T))
        
    for lam,phi in zip(Lam,Phi):
        h_next = pl.subs(h,x,phi)
        Lh = Lh + (h_next - h) * lam
    
    return Lh

def ctsSDPData(sys,c,h,X,MList,Eq = None):
    x = sys.x
    f = sys.f
    g = sys.g
    Lam = sys.Lam
    Phi = sys.Phi
    # Build the moment equations
    F = pl.polyarr([generator(x,mon,f,g,Lam,Phi) for mon in X])
    MPolys = pl.hstack([M.flatten() for M in MList])
    sdpPolys = pl.hstack([h,X,F,MPolys])
    if Eq is not None:
        sdpPolys =pl.hstack([sdpPolys,Eq])
    
    # Find U
    AllMonoms = sdpPolys.allmonomials()
    XMonoms = X.allmonomials()
    XMStringsSet = set([pl.monomialString(mon) for mon in XMonoms])

    
    AllMStringSet = set([pl.monomialString(mon) for mon in AllMonoms])
    
    UStringSet = AllMStringSet - XMStringsSet
    UMonoms = []
    for mon in AllMonoms:
        mStr = pl.monomialString(mon)
        if mStr in UStringSet:
            UMonoms.append(mon)
        
    U = pl.polyarr([pl.genpoly([(1,mon)]) for mon in UMonoms])

    Z = pl.hstack([X,U])
    
    # Now store the monomials for each entry of X and U
    XMonoms = [mon[0] for mon in X.monomials()]
    UMonoms = [mon[0] for mon in U.monomials()]
    ZMonoms = XMonoms + UMonoms
    
    # Construct the amtrices
    nX = len(X)
    nU = len(U)
    nZ = len(Z)
    
    C = np.zeros(nX)
    D = np.zeros(nU)
    H = np.zeros(nX)
    K = np.zeros(nU)
    A = np.zeros((nX,nX))
    B = np.zeros((nX,nU))

            
    
    for i,mi in enumerate(XMonoms):
        C[i] = c.coeff(mi)
        H[i] = h.coeff(mi)
        
    for i,mi in enumerate(UMonoms):
        D[i] = c.coeff(mi)
        K[i] = h.coeff(mi)
    
    for i,mi in enumerate(XMonoms):
        
        Fi = F[i]
        for j,mj in enumerate(XMonoms):
            A[i,j] = Fi.coeff(mj)
            
        for j,mj in enumerate(UMonoms):
            B[i,j] = Fi.coeff(mj)
            
    MarrList = []
    for M in MList:
        nM = len(M)
        Marr = np.zeros((nZ,nM,nM))

        for k,m in enumerate(ZMonoms):
            for i in range(nM):
                for j in range(nM):
                    Marr[k,i,j] = M[i,j].coeff(m)

        MarrList.append(Marr)

    if Eq is None:
        J = None
        L = None
    else:
        nE = len(Eq)
        J = np.zeros((nE,nX))
        L = np.zeros((nE,nU))

        for i,e in enumerate(Eq): 
            for j,mj in enumerate(XMonoms):
                J[i,j] = e.coeff(mj)
                
            for j,mj in enumerate(UMonoms):
                L[i,j] = e.coeff(mj)

        
    return U,A,B,C,D,H,K,MarrList,J,L

class jumpDiffusion:
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
    
Minimize = -1
Maximize = 1
Both = 0

randSolve = 'randSolve'
Euler = 'Euler'
Pseudospectral = 'Pseudospectral'

class objective:
    """
    obj = objective(integrand=pl.constant(0),terminal=pl.constant(0),
                    horizon=np.inf,objectiveType=Minimize)

    Creates an objective function. 
    """
    def __init__(self,integrand=pl.constant(0),terminal=pl.constant(0),
                 horizon=np.inf,objectiveType=Minimize):
        self.c = integrand
        self.h = terminal
        self.horizon = horizon
        self.objectiveType = objectiveType

def buildProbData(sys,obj,X,MList,Eq,Xinit,integrator):
    x = sys.x
    f = sys.f
    g = sys.g
    c = obj.c
    h = obj.h

    U,A,B,C,D,H,K,MarrList,J,L = ctsSDPData(sys,c,h,X,MList,Eq)

    nX = len(X)
    nU = len(U)

    horizon = obj.horizon

    if isinstance(horizon,np.ndarray):
        infHorizon = False
    elif horizon < np.inf:
        infHorizon = False
    else:
        infHorizon = True

    if infHorizon:
        XVar = cvx.Variable(nX)
        UVar = cvx.Variable(nU)

        constraints = [XVar[0,0] == 1.]

        constraints.append(A*XVar + B*UVar == 0)
        ZVar = cvx.vstack(XVar,UVar)

        for Marr in MarrList:
            MProd = [Mi * Zi for Mi,Zi in zip(Marr,ZVar)]
            MVar = reduce(lambda A,B : A + B,MProd)
            constraints.append(MVar >> 0)

        if J is not None:
            # If we have equalities
            constraints.append(J * XVar + L * UVar == 0)
        
        cost = H * XVar + K * UVar

    elif integrator == Euler:
        NumSteps = len(horizon)-1
        XVar = cvx.Variable(NumSteps+1,len(X))
        UVar = cvx.Variable(NumSteps+1,len(U)) 

        constraints = [XVar[0,:] == Xinit.reshape((1,len(Xinit)))]
        
        cost = H * XVar[-1,:].T + K * UVar[-1,:].T

        for k in range(NumSteps+1):
            xVar = XVar[k,:]
            uVar = UVar[k,:]
            zVar = [xv for xv in xVar] + [uv for uv in uVar]
            for M in MarrList:
                MProd = [Mi*zi for Mi,zi in zip(M,zVar)]
                MVar = reduce(lambda A,B : A+B,MProd)
                constraints.append(MVar >> 0)

            if k < NumSteps:
                xNextVar = XVar[k+1,:]
                dt = horizon[k+1] - horizon[k]
                
                constraints.append(dt * (A*xVar.T + B*uVar.T) + xVar.T \
                                   == xNextVar.T)
    
                cost += dt * (C * xVar.T + D * uVar.T)

            if J is not None:
                # If we have equalities
                constraints.append(J * XVar[k,:].T + L * UVar[k,:].T == 0)

    elif integrator == Pseudospectral:
        NumSteps = len(horizon)

        ti = horizon[0]
        tf = horizon[-1]
        scale = GLScale(ti,tf)

        Nodes,Weights = GLNodesAndWeights(NumSteps)
        Deriv = GLDerivativeMatrix(NumSteps)

        XVar = cvx.Variable(NumSteps,len(X))
        UVar = cvx.Variable(NumSteps,len(U))

        XDot = Deriv * XVar


        constraints = [XVar[0,:] == Xinit.reshape((1,len(Xinit)))]
        cost = H * XVar[-1,:].T + K * UVar[-1,:].T

        for k in range(NumSteps):
            xVar = XVar[k,:]
            uVar = UVar[k,:]
            zVar = [xv for xv in xVar] + [uv for uv in uVar]
            for M in MarrList:
                MProd = [Mi*zi for Mi,zi in zip(M,zVar)]
                MVar = reduce(lambda A,B : A+B,MProd)
                constraints.append(MVar >> 0)

            constraints.append(scale *  (A*xVar.T + B*uVar.T) == XDot[k,:].T)
            cost += Weights[k] * (C * xVar.T + D * uVar.T)

            if J is not None:
                # If we have equalities
                constraints.append(J * XVar[k,:].T + L * UVar[k,:].T == 0)

    return cost,constraints,XVar,UVar,U
    
        
def sdpBounds(sys,obj,X,MList,Eq=None,Xinit=None,solver=cvx.CVXOPT,integrator=Euler,
              numiter=100,smoothing_factor=1.,verbose=False):
    """
    sol = sdpBounds(sys,obj,X,MList,Eq=None,Xinit=None,solver=cvx.CVXOPT,integrator='Euler')

    Arugments:
    sys - A jumpDiffusion object
    obj - An objective object
    X - A polyarr containing the monomials of interest
    MList - A list of square polyarr matrices representing LMI constraints
    Xinit - An initial condition. Only required if obj is a finite-horizon objective
    solver - The SDP solver to be used by cvxpy. Optional.
    integrator - integrator for finite origons either 'Euler' or 'Pseudospectral'

    Returns:
    sol - A dictionary of the form

    sol = {'Value_Max' : Maximum Objective Value,
           'Value_Min' : Minimum Objective Value,
           'X_Max' : Maximizing X Solution,
           'X_Min' : Minimizing X Solution,
           'U_Max' : Maximizing U Solution,
           'U_Min' : Minimizing U Solution
           'U_Monomials' : polyarr of monomials in U}

    If obj specificies a maximization problem, then all of the entries corresponding 
    to minimization have values None.

    Similarly, if obj specifies a minimization, then all entries corresponding to 
    maximization have values None.

    If obj specifies both maximization and minimization, then all are defined
    
    """
    # I will need to check but likely this new addition broke some of the older examples

    cost,constraints,XVar,UVar,U = buildProbData(sys,obj,X,MList,Eq,Xinit,integrator)
    bounds = []
    xSol = []
    uSol = []

    solutionDict = {'Value_Max' : None,
                    'Value_Min' : None,
                    'X_Max' : None,
                    'X_Min' : None,
                    'U_Max' : None,
                    'U_Min' : None,
                    'U_Monomials' : U}
    
    if obj.objectiveType <= 0:
        minObj = cvx.Minimize(cost)

        minProb = cvx.Problem(minObj,constraints)
        if solver == randSolve:
            Hist = rs.randSolve(minProb,numiter=numiter,
                                returnSeq=False,jit=True,
                                smoothing_factor=smoothing_factor,
                                verbose=True)
            solutionDict['Value_Min'] = minProb.objective.value
        else:
            solutionDict['Value_Min'] = minProb.solve(solver=solver,max_iters=numiter,verbose=verbose)
        solutionDict['X_Min'] = XVar.value
        solutionDict['U_Min'] = UVar.value
        
    if obj.objectiveType >= 0:
        maxObj = cvx.Maximize(cost)
        maxProb = cvx.Problem(maxObj,constraints)
        if solver == randSolve:
            Hist = rs.randSolve(maxProb,numiter=numiter,
                                returnSeq=False,jit=True,
                                smoothing_factor=smoothing_factor,
                                verbose=True)
            solutionDict['Value_Max'] = maxProb.objective.value
        else:
            solutionDict['Value_Max'] = maxProb.solve(solver=solver,max_iters=numiter,verbose=verbose)
        solutionDict['X_Max'] = XVar.value
        solutionDict['U_Max'] = UVar.value


    return solutionDict
        

###### Pseudospectral Helpers #####

def GLNodesAndWeights(NumSteps):
    LegPoly = sfun.legendre(NumSteps-1)
    LegPoly_deriv = LegPoly.deriv()
    PSNodes = np.sort(np.hstack((-1,LegPoly_deriv.r,1)))
    PSWeights = 2./(NumSteps*(NumSteps-1)*LegPoly(PSNodes)**2.)
    return PSNodes,PSWeights

def GLTime(ti,tf,NumSteps):
    """
    Time = GLTime(ti,tf,NumSteps)

    Creates an array of time nodes over [ti,tf] spaced using a 
    Gauss-Lobato pattern.
    """
    PSNodes,_ = GLNodesAndWeights(NumSteps)
    a =.5 * (tf - ti)
    b = .5 * (ti+tf)
    return a * PSNodes + b

def GLScale(ti,tf):
    return .5 * (tf-ti)


def LagPolys(NumSteps):
    LagPolyList = []
    PSNodes,_ = GLNodesAndWeights(NumSteps)
    for k in range(NumSteps):
        roots = np.hstack((PSNodes[:k],PSNodes[k+1:]))
        LagPoly = np.poly1d(roots,r=True)
        LagPoly = LagPoly / LagPoly(PSNodes[k])
        LagPolyList.append(LagPoly)

    return LagPolyList

def GLDerivativeMatrix(NumSteps):
    D = np.zeros((NumSteps,NumSteps))
    PSNodes,_ = GLNodesAndWeights(NumSteps)
    LagPolyList = LagPolys(NumSteps)
    for k in range(NumSteps):
        LagPoly = LagPolyList[k]
        LagPoly_deriv = LagPoly.deriv()
        D[:,k] = LagPoly_deriv(PSNodes)


        
    return D

def LagInterp(Time,X):
    """
    Use the Lagrange interpolating polynomial to calculate values of X at intermediate times.
    It is assumed that Time[0] and Time[-1] correspond to the true initial and final times.
    """

    ti = Time[0]
    tf = Time[-1]
    a = 2./(tf - ti)
    b = (tf + ti) / (ti - tf)

    Tau = a * Time + b

    NumSteps,nX = X.shape
    LagPolyList = LagPolys(NumSteps)

    X_interp = np.zeros((len(Tau),nX))

    for k in range(NumSteps):
        X_interp += np.outer(LagPolyList[k](Tau),X[k])

    return X_interp

    
#### Helpers #####
#### These things are to make the optimization process more automatic ####

def maximalState(sys,M):
    """
    
    X,F = maximalState(sys,M)
    
    Find the largest state vector such that all states and their corresponding
    generator are included in the monomials of M
    """
    
    x = sys.x
    
    f = sys.f
    g = sys.g
    Lam = sys.Lam
    Phi = sys.Phi
    
    XAll = pl.pureMonomials(x,M)
    
    XContained = []
    FContained = []
    
    MatMons = M.monomialArr()
    
    for xi in XAll:
        fi = generator(x,xi,f,g,Lam,Phi)
        MonArr = pl.polyarr([fi]).monomialArr()
        
        if len(pl.residualMonomials(MonArr,MatMons)) == 0:
            XContained.append(xi)
            FContained.append(fi)
            
            
    X = pl.polyarr(XContained)
    F = pl.polyarr(FContained)
    return X,F

def maximalEqualityConstraint(a,M,Z=None):
    """
    Eq = maximalEqualityConstraint(a,M,Z=None)
    
    Assume that a == 0 is a vector of algebraic constraints.
    
    returns maximal nacollection of vectors p * a 
    such that all monomials in the collection are contained in M
    
    If Z is given, the p are chosen from Z, otherwise, they are 
    chosen from the monomials of M
    
    """
    if Z is None:
        Z = M.monomialArr()
        
    AAll = pl.hstack([a*p for p in Z])
    
    AContained = []
    
    MatMons = M.monomialArr()
    
    for p in AAll:
        if len(pl.residualMonomials(pl.polyarr([p]),MatMons)) == 0:
            AContained.append(p)
            
    return pl.polyarr(AContained)
