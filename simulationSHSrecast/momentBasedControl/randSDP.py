import numpy as np
import numpy.random as rnd
import cvxpy as cvx
from numba import jit
import scipy.linalg as la
import utils as ut

from IPython import embed


@jit
def projectEquality(y,a,b):
    if np.dot(a,y) == b:
        return y
    else:
        r = np.dot(a,y) - b
        lam = r / np.dot(a,a)
        z = y - lam * a
        return z

@jit
def projectInequality(y,a,b,overrelax=0.0):
    r = np.dot(a,y) - b
    if r <= 0:
        z = y
    else:
        lam = (r + overrelax )/ np.dot(a,a)
        z = y - lam * a

    return z

@jit
def matDim(n):
    return int(np.round((np.sqrt(1+8*n)-1)/2))

@jit
def symMat(z):
    n = len(z)
    d = matDim(n)
    
    M = np.zeros((d,d))
    
    count = 0 
    for i in range(d):
        for j in range(i,d):
            m = z[count]
            if i == j:
                M[i,j] = m
            elif i!=j:
                # Funny square root for consistency with SCS matrices
                M[i,j] = m/np.sqrt(2)
                M[j,i] = M[i,j]
                
            count += 1
    return M

@jit 
def quadVec(v):
    d = len(v)
    vVec = np.zeros(d*(d+1)/2)
    count = 0
    for i in range(d):
        vVec[count] = v[i]**2
        # Funny square root for consistency with SCS
        vVec[count+1:count+d-i] = np.sqrt(2)*v[i] * v[i+1:]
        count += d-i
    return vVec

@jit
def quadForm(z,v):
    vVec = quadVec(v)
    return np.dot(vVec,z)

@jit(nopython=True)
def randSDP(c,A=np.zeros((0,0)),b=np.zeros(0),F=np.zeros((0,0)),g=np.zeros(0),
            M = np.zeros((0,0)),p =np.zeros(0), dims = np.zeros(0,dtype=int),
            NumSteps=100, OvrPara = 0.,smoothing_factor = 1.):


    m,n = M.shape
    NumEq = len(b)
    NumIneq = len(g)
    NumLMI = len(dims)

    NumConstraints = NumEq + NumIneq + NumLMI

    if NumEq > 0:
        n = A.shape[1]
    elif NumIneq > 0:
        n = F.shape[1]
    elif NumLMI > 0:
        n = M.shape[1]


    lmiVarNum = []
    lmiOffsetList = [0]
    offsetList = [0]
    for lmi in range(NumLMI):
        d = dims[lmi]
        dLast = offsetList[-1]
        offsetList.append(dLast+d)
        mSize = d*(d+1)/2
        lmiVarNum.append(mSize)
        lastOffset = lmiOffsetList[-1]
        lmiOffsetList.append(lastOffset+mSize)


    dimV = np.sum(dims)
    bigV = np.zeros(dimV)
    for lmi in range(NumLMI):
        d = dims[lmi]
        sInd = offsetList[lmi]
        v = rnd.randn(d)
        v = v / np.linalg.norm(v)
        bigV[sInd:sInd+d] = v

    nextInd = -1

    
    # set the selection probability
    sel_prob = 0.95

    # Main iteration
    # constraintSeq = rnd.randint(NumConstraints,size=NumSteps)
    x = np.zeros(n)
    x_ave = np.zeros(n)
    for k in range(NumSteps):
        if nextInd < 0:
            ind = rnd.randint(NumConstraints)
        else:
            ind = nextInd

        # Use aggressive step sizes 
        eta = (1./(k+1.))**(.8)
        
        y = x - eta * c
        np.dot(c,x),np.dot(c,y)

        if ind < NumEq:
            aInd = A[ind]
            bInd = b[ind]
            x = projectEquality(y,aInd,bInd)
        elif ind < NumEq + NumIneq:
            aInd = F[ind-NumEq]
            bInd = g[ind-NumEq]

            ovr = OvrPara*eta * np.linalg.norm(aInd)

            
            x = projectInequality(y,aInd,bInd,ovr)

        elif ind < NumConstraints:
            mInd = ind - NumEq - NumIneq

            
            FLMI = M[lmiOffsetList[mInd]:lmiOffsetList[mInd+1]]
            gLMI = p[lmiOffsetList[mInd]:lmiOffsetList[mInd+1]]


            # Find what would be the value of M
            MVal = symMat(gLMI - np.dot(FLMI,y))
            # update an estimate of its most negative eigenvalue

            sInd = offsetList[mInd]
            LMISize = dims[mInd]
            v = bigV[sInd:sInd+LMISize]


            # Again, using very aggressive values
            v = v - (eta**(0.7)) * np.dot(MVal,v)
            v = v / np.linalg.norm(v)
            


            lam = np.dot(v,np.dot(MVal,v))
            if lam < -1e-1 and rnd.rand()<=sel_prob:
                # perform selection strategy as in Reinforcement Learning
                nextInd = ind
            else:
                nextInd = -1
                w = rnd.randn(LMISize)
                v = v + eta * w
                v = v / np.linalg.norm(v)

            bigV[sInd:sInd+LMISize] = v
                
            # Project onto this constraint

            vVec = quadVec(v)

            bInd = np.dot(vVec,gLMI)
            aInd = np.dot(vVec,FLMI)

            ovr = OvrPara*eta * np.linalg.norm(aInd)
                
            x = projectInequality(y,aInd,bInd,ovr)

        else:
            x = y    

        if np.linalg.norm(x-y) > 1e-6:
            activeConstraint = 1
        else:
            activeConstraint = 0

        x_ave = smoothing_factor * x + (1-smoothing_factor) * x_ave
    return x_ave

def getMatrices(prob):
    data = prob.get_problem_data(cvx.SCS)

    BigA = np.array(data['A'].todense())
    Bigb = data['b']
    c = data['c']

    dims = data['dims']

    NumEq = dims['f']
    NumIneq = dims['l']
    NumLMI = len(dims['s'])


    A = BigA[:NumEq]
    b = Bigb[:NumEq]


    F = BigA[NumEq:NumEq+NumIneq]
    g = Bigb[NumEq:NumEq+NumIneq]

    M = BigA[NumEq+NumIneq:]
    p = Bigb[NumEq+NumIneq:]

    return c,A,b,F,g,M,p,np.array(dims['s'],dtype=int)
    
#@jit
def eliminateEqualities(c,A,b,F,g,M,p,dims):
    if A.shape[0] == 0:
        n = len(c)
        return (c,A,b,F,g,M,p,dims),(np.zeros(n),np.eye(n))

    tol = 1e-10
    U,S,Vt = la.svd(A)

    rank = len(S>tol)
    SigInv = np.diag(1./S[:rank])

    U1 = U[:,:rank]
    U2 = U[:,rank:]

    if U2.shape[1] > 0:
        if la.norm(np.dot(U2.T,b)) > tol:
            return

    V1 = Vt[:rank].T
    W = Vt[rank:].T

    # Compute the least squares solution
    h = la.lstsq(A,b)[0]
    # This could be computed via svd, but this is better conditioned.
    

    cNew = np.dot(c,W)
    FNew = np.dot(F,W)
    gNew = g - np.dot(F,h)

    MNew = np.dot(M,W)
    pNew = p - np.dot(M,h)

    n,m = W.shape 
    
    ANew = np.zeros((0,m))
    bNew = np.zeros((0,))

    MatsNew = (cNew,ANew,bNew,FNew,gNew,MNew,pNew,dims)
    Trans = (h,W)
 
    return MatsNew,Trans


def randSolve(prob,numiter=100,returnSeq=True, setOvrPara = 0.,smoothing_factor=1.,
              jit=False,verbose=False):
    if verbose:
        print 'Finding the matrices'
    Mats = getMatrices(prob)
    if verbose:
        print 'Eliminate Equalities'
    MatsAndTrans = eliminateEqualities(*Mats)
    if MatsAndTrans is None:
        print 'Infeasible Equality Constraints'
        return
    else:
        Mats = MatsAndTrans[0]
        h,W =MatsAndTrans[1]

    if verbose:
        print 'Main Loop'
    if jit and not returnSeq:
        # In jit mode we do not return history
        # The problem is that the history is BIG
        # In no jit mode you have a choice of history or no history
        # But with jit, you cannot leave the return size as variable.
        Z = randSDP(*Mats, NumSteps=numiter, OvrPara = setOvrPara)
    else:
        Z = randSDP_nojit(*Mats, NumSteps=numiter, OvrPara = setOvrPara,smoothing_factor=smoothing_factor,history=returnSeq)
        
        
    n,m = W.shape
    if returnSeq:
        X = np.zeros((len(Z),n))
        for k in range(len(Z)):
            X[k] = h + np.dot(W,Z[k])

        x = X[-1]
    else:
        x = h + np.dot(W,Z)
            

    if verbose:
        print 'Stuff values'
    # Stuff the values into the original problem
    oc,cc = prob.canonicalize()
    scs = cvx.problems.solvers.scs_intf.SCS()
    sdata = scs.get_sym_data(oc,cc,prob._cached_data)
    prob._save_values(x,prob.variables(),sdata.var_offsets)

    if returnSeq:
        return X
    else:
        return prob.objective.value


def randSDP_nojit(c,A=np.zeros((0,0)),b=np.zeros(0),F=np.zeros((0,0)),g=np.zeros(0),
                  M = np.zeros((0,0)),p =np.zeros(0), dims = [],
                  NumSteps=100, OvrPara = 0.,smoothing_factor = 1.,history=True):

    
    NumEq = len(b)
    NumIneq = len(g)
    NumLMI = len(dims)

    NumConstraints = NumEq + NumIneq + NumLMI

    if NumEq > 0:
        n = A.shape[1]
    elif NumIneq > 0:
        n = F.shape[1]
    elif NumLMI > 0:
        n = M.shape[1]


    lmiVarNum = np.zeros(NumLMI,dtype=int)
    for lmi in range(NumLMI):
        d = dims[lmi]
        lmiVarNum[lmi] = d*(d+1)/2

    
    lmiOffsetList = np.cumsum(np.hstack([0,lmiVarNum]))

    dimV = np.sum(dims)
    bigV = np.zeros(dimV)
    offsetList = np.hstack([0,np.cumsum(dims)])
    for lmi in range(NumLMI):
        d = dims[lmi]
        sInd = offsetList[lmi]
        v = rnd.randn(d)
        v = v / np.linalg.norm(v)
        bigV[sInd:sInd+d] = v

    nextInd = -1

    # set the selection probability
    sel_prob = 0.95

   
    # Main iteration
    constraintSeq = rnd.choice(range(NumConstraints),NumSteps)
    x = np.zeros(n)
    x_ave = np.zeros(n)
    if history:
        X = np.zeros((NumSteps,n))

    else:
        X = np.zeros((0,n))
        
    for k in range(NumSteps):
        if nextInd < 0:
            ind = constraintSeq[k]
        else:
            ind = nextInd

        # Use aggressive step sizes
        eta = np.max([(1./(k+1.)),1e-3])
        
        #eta = (1.0e-2) *  (1.0e5/(k+1.0e5))
        #eta = 10./(k+10.)

        eta = (1./(k+1.))**(.75)
        
        y = x - eta * c
        np.dot(c,x),np.dot(c,y)

        if ind < NumEq:
            aInd = A[ind]
            bInd = b[ind]
            x = projectEquality(y,aInd,bInd)
        elif ind < NumEq + NumIneq:
            aInd = F[ind-NumEq]
            bInd = g[ind-NumEq]

            ovr = OvrPara*eta * np.linalg.norm(aInd)

            
            x = projectInequality(y,aInd,bInd,ovr)

        elif ind < NumConstraints:
            mInd = ind - NumEq - NumIneq

            
            FLMI = M[lmiOffsetList[mInd]:lmiOffsetList[mInd+1]]
            gLMI = p[lmiOffsetList[mInd]:lmiOffsetList[mInd+1]]


            # Find what would be the value of M
            MVal = symMat(gLMI - np.dot(FLMI,y))
            # update an estimate of its most negative eigenvalue

            sInd = offsetList[mInd]
            LMISize = dims[mInd]
            v = bigV[sInd:sInd+LMISize]


            # Again, using very aggressive values
            v = v - (eta**(0.7)) * np.dot(MVal,v)
            v = v / np.linalg.norm(v)
            


            lam = np.dot(v,np.dot(MVal,v))
            if lam < -1e-1 and rnd.uniform(0,1,1)<=sel_prob:
                # perform selection strategy as in Reinforcement Learning
                nextInd = ind
            else:
                nextInd = -1
                w = rnd.randn(LMISize)
                v = v + eta * w
                v = v / np.linalg.norm(v)

            bigV[sInd:sInd+LMISize] = v
                
            # Project onto this constraint

            vVec = quadVec(v)

            bInd = np.dot(vVec,gLMI)
            aInd = np.dot(vVec,FLMI)

            ovr = OvrPara*eta * np.linalg.norm(aInd)
                
            x = projectInequality(y,aInd,bInd,ovr)

        else:
            x = y    

        if np.linalg.norm(x-y) > 1e-6:
            activeConstraint = 1
        else:
            activeConstraint = 0

        x_ave = smoothing_factor * x + (1-smoothing_factor) * x_ave
        if history:
            X[k] = x_ave

    if history:
        return X
    else:
        return x_ave

