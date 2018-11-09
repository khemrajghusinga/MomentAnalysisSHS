import numpy as np
import sympy as sym
import scipy.linalg as la

def controllabilityMatrix(A,B):
    n,p = B.shape

    Mat = np.array(B,copy=True)

    Cont = np.zeros((n,n*p))

    for k in range(n):
        Cont[:,k*p:(k+1)*p] = Mat
        Mat = np.dot(A,Mat)

    return Cont

def isControllable(A,B):
    Cont = controllabilityMatrix(A,B)
    Value = (np.linalg.matrix_rank(Cont) == len(A))
    return Value


def observabilityMatrix(A,C):
    return controllabilityMatrix(A.T,C.T).T

def minreal(Sys,tol=1e-8):
    A,B,C,D = Sys
    Cont = controllabilityMatrix(A,B)
    Obs = observabilityMatrix(A,C)

    H = np.dot(Obs,Cont)
    Hs = np.dot(Obs,np.dot(A,Cont))

    U,S,Vh = la.svd(H)

    if len(np.argwhere(S>tol)) > 0:
        rank = np.argwhere(S>tol)[-1,0] + 1

        rtS = np.diag(np.sqrt(S[:rank]))
        ObsMin = np.dot(U[:,:rank],rtS)
        ContMin = np.dot(rtS,Vh[:rank])
    
        Cmin = ObsMin[:len(C)]
        Bmin = ContMin[:,:B.shape[1]]

        Amin = la.lstsq(ObsMin,la.lstsq(ContMin.T,Hs.T)[0].T)[0]

    else:
        Amin = np.zeros((1,1))
        Bmin = np.zeros((1,1))
        Cmin = np.zeros((1,1))

    return Amin,Bmin,Cmin,D

def prodStateSpace(P1,P2):
    A1,B1,C1,D1 = P1
    A2,B2,C2,D2 = P2

    Atop = np.hstack((A1,np.dot(B1,C2)))
    Abot = np.hstack((np.zeros((len(A1),len(A2))),A2))
    A = np.vstack((Atop,Abot))

    B = np.vstack((np.dot(B1,D2),
                   B2))

    C = np.hstack((C1,np.dot(D1,C2)))
    D = np.dot(D1,D2)

    return A,B,C,D

def block_matrix(MBlock):
    Rows = [np.hstack(row) for row in MBlock]
    return np.vstack(Rows)

def dot(*args):
    M = reduce(lambda A,B : np.dot(A,B), args)
    return M

def cast_as_array(x):
    if isinstance(x,np.ndarray):
        return x
    else:
        return np.array(x)

def jacobian(F,x):
    """ 
    Computes: 
    J = dF / dx

    If x is a scalar, then J is just the derivative of each entry of F.

    If F is a scalar and x is a vector, then J is the gradient of F with respect to x. 

    If F and x are 1D vectors, then J is the standard Jacobian.

    More generally, the shape of J will be given by
    
    J.shape = (F.shape[0],...,F.shape[-1], x.shape[0],..., x.shape[-1])
    
    This assumes that F and x are scalars or arrays of symbolic variables.
    
    
    """

    Farr = cast_as_array(F)
    xarr = cast_as_array(x)

    Fflat = Farr.flatten()
    xflat = xarr.flatten()

    nF = len(Fflat)
    nx = len(xflat)
    # a matrix to hold the derivatives
    Mat = np.zeros((nF,nx),dtype=object)

    for i in range(nF):
        for j in range(nx):
            Mat[i,j] = sym.diff(Fflat[i],xflat[j])

    Jac = np.reshape(Mat, Farr.shape + xarr.shape)
    return Jac



def simplify(x):
    if isinstance(x,np.ndarray):
        xflat = x.flatten()
        nX = len(xflat)
        x_simp_list = [sym.simplify(expr) for expr in xflat]
        x_simp = np.reshape(x_simp_list,x.shape)
        return x_simp
    else:
        return sym.simplify(x)


def subvar(f,x,xVal):
    if isinstance(x,np.ndarray):
        xflat = x.flatten()
        xValflat = xVal.flatten()
    else:
        xflat = [x]
        xValflat = [xVal]

    nX = len(xflat)
    
    if isinstance(f,np.ndarray):
        fflat = f.flatten()
        nF = len(fflat)

        fSubflat = np.zeros(nF,dtype=object)
        
        for i in range(nF):
            fCur = fflat[i]
            for (xCur,xValCur) in zip(xflat,xValflat):
                fCur = fCur.subs(xCur,xValCur)

            fSubflat[i] = fCur

        return np.reshape(fSubflat,f.shape)
    else:
        for i in range(nX):
            xCur = xflat[i]
            xValCur = xValflat[i]
            if i==0:
                fCur = f.subs(xCur,xValCur)
            else:
                fCur = fCur.subs(xCur,xValCur)

        return fCur

def subs(f,x,xVal):
    if isinstance(x,tuple):
        f_sub = None
        for v,vVal in zip(x,xVal):
            if f_sub is None:
                f_sub = subvar(f,v,vVal)
            else:
                f_sub = subvar(f_sub,v,vVal)
    else:
        f_sub = subvar(f,x,xVal)

    return f_sub
        
        
def arg_to_flat_tuple(argTup):
    """
    Takes in a tuple of arguments. If any are np.ndarrays, they will 
    be flattended and added to the tuple
    """
    argList = []
    for variable in argTup:
        if isinstance(variable,np.ndarray):
            argList.extend(variable.flatten())
        else:
            argList.append(variable)
    return tuple(argList)
    

def functify(expr, args,shape=()):
    """
    Usage: 
    name = functify(expr,args)
    This creates a function of the form
    expr = name(args)

    For more information, see
    https://www.youtube.com/watch?v=99JS6ym5FNE
    """
    if isinstance(args,tuple):
        argTup = args
    else:
        argTup = (args,)

    FuncData = {'nvar': len(argTup),
                'shape': shape,
                'squeeze': False,
                'flatten': False}

    if shape is not ():
        squeezeVec = False
    else:
        squeezeVec = True

    flatTup = arg_to_flat_tuple(argTup)
    if isinstance(expr,np.ndarray):
        FuncData['shape'] = expr.shape
        if (len(expr.shape) < 2) and squeezeVec:
            FuncData['squeeze'] = squeezeVec
            exprCast = sym.Matrix(expr)
        elif (len(expr.shape) > 2) or not squeezeVec:
            FuncData['flatten'] = True
            exprCast = sym.Matrix(expr.flatten())
        else:
            exprCast = sym.Matrix(expr)

        mods = [{'ImmutableMatrix': np.array}, 'numpy']
        func = sym.lambdify(flatTup,exprCast,modules=mods)
    else:
        func = sym.lambdify(flatTup,expr)

    def foo(*new_arg):
        new_flatTup = arg_to_flat_tuple(new_arg)
        result = func(*new_flatTup)
        if FuncData['squeeze']:
            return result.squeeze()
        elif FuncData['flatten']:
            return np.reshape(result,FuncData['shape'])
        else:
            return result

    return foo

