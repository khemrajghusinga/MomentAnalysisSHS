import numpy as np
import utils as ut
import scipy.linalg as la
import numbers

def monomialString(mon):
    if mon == constantMonomial:
        mStr = ''
            
    else:
        mStr = ''
        for v,d,f in mon:
            if f == 'pow':
                mStr += '%s^%d' % (v,d)
            else:
                mStr += '%s(%d%s)' % (f,d,v)
                    
    return mStr

class univariatepoly:
    """
    Internal univariate class
    """
    
    def __init__(self,plist):
        self.plist = plist
        self.casttoconstants()
        self.sortmonomials()
        self.absorbconstants()
        self.simplifyproducts()
        self.sortsum()
        self.compresssum()
  
    def casttoconstants(self):
        newList = []
        for c,mon in self.plist:
            newMon = []
            cNew = c
            for v,d,f in mon:
                if (d==0):
                    if f == 'pow':
                        newMon.extend(constantMonomial)
                    elif f == 'cos':
                        newMon.extend(constantMonomial)
                    elif f == 'sin':
                        newMon = constantMonomial
                        cNew = 0
                        break
                else:
                    newMon.append((v,d,f))
                    
            if len(newMon) == 0:
                newMon = constantMonomial
                
            newList.append((cNew,newMon))
            
        self.plist = newList
                        
                        
    def sortmonomials(self):
        for c,mon in self.plist:
            mon.sort(key = lambda tup : tup[0])
                    
            
    def absorbconstants(self):
        newList = []
        for c,mon in self.plist:
            v0,d0,f0 = mon[0]
            newMon = [mon[0]]
            for v,d,f in mon[1:]:
                if (v == v0) and (v != ''):
                    # This ensures that the constant monomial is only appended once
                    newMon.append((v,d,f))
                else:
                    newMon = [(v,d,f)]
                    v0 = v
                    d0 = d
                    f0 = f
                    
            newList.append((c,newMon))
        self.plist = newList
        
    def simplifyproducts(self):
        newList = []
        for c,mon in self.plist:
            v0,d0,f0 = mon[0]
            # Assuming all fs in monomial are '', 'pow' or ('sin'/'cos')
            if f0 == '':
                simpleProd = [(c,mon)]
            elif f0 == 'pow':
                dNew = np.sum([d for v,d,f in mon])
                newMon = [(v0,dNew,f0)]
                simpleProd = [(c,newMon)]
            else: 
                if len(mon) == 1:
                    simpleProd = [(c,mon)]
                else:
                    v1,d1,f1 = mon[1]
                    vVar = var(v0)
                    vSum = (d0+d1)*vVar
                    vDif = (d0-d1)*vVar
                    if f0 == 'cos' and f1 == 'cos':
                        p1 = .5 * cos(vSum)
                        p2 = .5 * cos(vDif)
                        
                    if f0 == 'cos' and f1 == 'sin':
                        p1 = .5 * sin(vSum)
                        p2 = -.5 * sin(vDif)
                        
                    if f0 == 'sin' and f1 == 'cos':
                        p1 = .5 * sin(vSum)
                        p2 = .5 * sin(vDif)
                        
                    if f0 == 'sin' and f1 == 'sin':
                        p1 = -.5 * cos(vSum)
                        p2 = .5 * cos(vDif)
                        
                    headPoly = c*(p1+p2)
                    if len(mon) == 2:
                        simpleProd = headPoly.plist
                        
                    else:
                        tailPoly = univariatepoly([(1,mon[2:])])
                        simpleProd = (headPoly*tailPoly).plist
                    
            newList.extend(simpleProd)
            
        self.plist = newList
        
    def sortsum(self):
        self.plist.sort(key=lambda tup : monomialString(tup[1]))
        
 
    
    def compresssum(self):
        newList = []
        
        cCur,monCur = self.plist[0]
        
        for c,mon in self.plist[1:]:
            if mon == monCur:
                cCur += c
            else:
                if cCur != 0:
                    newList.append((cCur,monCur))
                cCur = c
                monCur = mon
                
        if cCur != 0:
            newList.append((cCur,monCur))
            
        if len(newList) == 0:
            newList = [(0,constantMonomial)]
        self.plist = newList
                
    def __str__(self):
        strList = []
        
        for c,mon in self.plist:
            Re = np.real(c)
            Im = np.imag(c)
            
            if Im == 0:
                cStr = '%g' % Re
            elif Re == 0:
                cStr = '%gj' % Im
            else:
                cStr = '(%g + %gj)' % (Re,Im)
                
            strList.append(cStr + monomialString(mon))
            
        return ' + '.join(strList)
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self,other):
        if isinstance(other,numbers.Number):
            other = constant(other)
            
        return univariatepoly(self.plist+other.plist)
    
    def __neg__(self):
        return self.__mul__(-1)
    
    def __sub__(self,other):
        return self.__add__(-other)
    
    def __rsum__(self,other):
        return self.__neg__()+other
    
    def __mul__(self,other):
        if isinstance(other,numbers.Number):
            other = constant(other)
            
        plist = []
        
        for coef,mon in self.plist:
            for newCoef,newMon in other.plist:
                plist.append((coef*newCoef,mon+newMon))
        
        return univariatepoly(plist)
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    
    def __div__(self,other):
        if isinstance(other,numbers.Number):
            return self.__mul__(1./other)
    
    def diff(self,dv):
        newList = []
        for c,mon in self.plist:
            # Univariate monomial lists will always have length 1
            v,d,f = mon[0]
            if f == 'pow':
                newList.append((c*d,[(v,d-1,f)]))
            elif f == 'sin':
                newList.append((c*d,[(v,d,'cos')]))
            elif f == 'cos':
                newList.append((-c*d,[(v,d,'sin')]))
                
        return univariatepoly(newList)
                
                    

class genpoly:
    def __init__(self,plist):
        self.plist = plist
        self.sortmonomials()
        self.simplifyproducts()
        self.sortsum()
        self.compresssum()
        self.sortmonomials()
        self.makemonomialdict()
        self.polyType = self.__class__.__name__

    def sortmonomials(self):
        for c,mon in self.plist:
            mon.sort(key = lambda tup : tup[0])
            
    def simplifyproducts(self):
        newList = []
        for c,mon in self.plist:
            varGroups = []
            gp = []
            vCur,dCur,fCur = mon[0]
            gp.append((vCur,dCur,fCur))
            
            for v,d,f in mon[1:]:
                if (v == vCur) and (v != ''):
                    gp.append((v,d,f))
                else:
                    if vCur != '':
                        varGroups.append(gp)
                    vCur = v
                    dCur = d
                    fCur = f
                    
                    gp = [(vCur,dCur,fCur)]
                    
            varGroups.append(gp)
            UVPList = []
            
            for k,gp in enumerate(varGroups):
                if k == 0:
                    uvp = univariatepoly([(c,gp)])
                else:
                    uvp = univariatepoly([(1,gp)])

                UVPList.append(uvp)
                
            mixedList = self.multiplymixedunivariatelist(UVPList)
            newList.extend(mixedList)
                
        self.plist = newList
                
    def multiplymixedunivariatelist(self,UVPS):
        head = UVPS[0].plist
        if len(UVPS) == 1:
            return head
        else:
            prodList = []
            tail = self.multiplymixedunivariatelist(UVPS[1:])
            for ch,mh in head:
                for ct,mt in tail:
                    if (mh == constantMonomial):
                        msum = mt
                    elif (mt == constantMonomial):
                        msum = mh
                        
                    else:
                        msum = mt+mh
                        
                    prodList.append((ch*ct,msum))
                    
            return prodList
        
    def sortsum(self):
        self.plist.sort(key=lambda tup : monomialString(tup[1]))
        
    
    def compresssum(self):
        newList = []
        
        cCur,monCur = self.plist[0]
        
        for c,mon in self.plist[1:]:
            if mon == monCur:
                cCur += c
            else:
                if cCur != 0:
                    newList.append((cCur,monCur))
                cCur = c
                monCur = mon
                
        if cCur != 0:
            newList.append((cCur,monCur))
            
        if len(newList) == 0:
            newList = [(0,constantMonomial)]
        self.plist = newList
        
    def makemonomialdict(self):
        self.Dict = {monomialString(m) : c for c,m in self.plist}

    def __str__(self):
        strList = []
        
        for c,mon in self.plist:
            Re = np.real(c)
            Im = np.imag(c)
            
            if Im == 0:
                cStr = '%g' % Re
            elif Re == 0:
                cStr = '%gj' % Im
            else:
                cStr = '(%g + %gj)' % (Re,Im)
                
            strList.append(cStr + monomialString(mon))
            
        return ' + '.join(strList)
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self,other):
        if isinstance(other,numbers.Number):
            other = constant(other)
        if isinstance(other,genpoly):
            return genpoly(self.plist+other.plist)
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __neg__(self):
        return self.__mul__(-1)
    
    def __sub__(self,other):
        return self.__add__(-other)
    
    def __rsub__(self,other):
        return self.__neg__()+other
                
    
    def __mul__(self,other):
        if isinstance(other,polyarr):
            return other.__mul__(self)

        if isinstance(other,numbers.Number):
            other = constant(other)
            
        
        plist = []
        
        for coef,mon in self.plist:
            for newCoef,newMon in other.plist:
                plist.append((coef*newCoef,mon+newMon))
        
        return genpoly(plist)
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __div__(self,other):
        if isinstance(other,numbers.Number):
            return self.__mul__(1./other)
        
    def __pow__(self,p):
        if isinstance(p,int):
            newPoly = constant(1)
            for k in range(p):
                newPoly = self.__mul__(newPoly)
            return newPoly

        elif isinstance(p,np.ndarray):
            pflat = p.flatten()
            flatArray = []
            for pv in pflat:
                pPow = self.__pow__(int(pv))
                flatArray.append(pPow)

            Arr = np.reshape(flatArray,p.shape)
            return polyarr(Arr)
    
    def diff(self,dv):
        newPoly = constant(0)
        for c,mon in self.plist:
            indMon = []
            depPoly = None
            for v,d,f in mon: 
                if v != dv.name:
                    indMon.append((v,d,f))
                else:
                    
                    depPoly = univariatepoly([(1,[(v,d,f)])]).diff(dv)
                    depPoly = genpoly(depPoly.plist)
                    
            if depPoly is not None:
                if len(indMon) == 0:
                    indMon = constantMonomial
                newPoly = newPoly + depPoly * genpoly([(c,indMon)])
    
        return newPoly 
    
    def integral(self,iv,a,b):
        # Currently just for definite integrals with numerical bounds
        newList = []
        for c,mon in self.plist:
            ivInMon = False
        
            indMon = []
            for v,d,f in mon:
                if v != iv.name:
                    indMon.append((v,d,f))
                else:
                    ivInMon  = True
                    if f == 'pow':
                        cMul = (b**(d+1)-a**(d+1)) / float(d+1)
                    elif f == 'cos':
                        cMul = (np.sin(b*d)-np.sin(a*d)) / float(d)
                    elif f == 'sin':
                        cMul = -(np.cos(b*d)-np.cos(b*d)) / float(d)
                        
                        
            if not ivInMon:
                cMul = b-a
                
            if len(indMon) == 0:
                indMon = constantMonomial
                        
            newList.append((cMul*c,indMon))
    
        return genpoly(newList)

    def sub(self,vx,x):
        newPoly = constant(0)

        for c,mon in self.plist:
            indMon = []
            depVal = 1
            for v,d,f in mon:
                if v != vx.name:
                    indMon.append((v,d,f))
                else:
                    if f == 'pow':
                        depVal = x ** d
                    elif f == 'sin':
                        depVal = sin(d*x)
                    elif f == 'cos':
                        depVal = cos(d*x)
            if len(indMon) == 0:
                indMon = constantMonomial
            newPoly = newPoly + depVal * genpoly([(c,indMon)])
        return newPoly  

    def val(self,valueDict):
        curVal = 0
        for c,mon in self.plist:
            monVal = 1
            for v,d,f in mon:
                if f == 'pow':
                    monVal *= valueDict[v]**d
                elif f == 'cos':
                    monVal *= np.cos(d*valueDict[v])
                elif f == 'sin':
                    monVal *= np.sin(d*valueDict[v])
                
            curVal += c * monVal
        return curVal
 
    
    def monomials(self):
        monList = [mon for c,mon in self.plist]
        monList.sort(key = lambda tup : monomialString(tup))
        newList = []
        mCur = monList[0]
        for m in monList[1:]:
            if m != mCur:
                newList.append(mCur)
                mCur = m
                
        newList.append(mCur)
        
        return newList
    
    def variables(self):
        vList = []
        for c,mon in self.plist:
            for v,d,f in mon:
                if v != '':
                    vList.append(v)
        vSet = set(vList)
        vList = list(vSet)
        vList.sort()
        return [var(v) for v in vList]
    
    def coeff(self,mon):
        mStr = monomialString(mon)
        if mStr in self.Dict.keys():
            return self.Dict[mStr]
        else:
            return 0
        
    def isZero(self):
        for c,mon in self.plist:
            if c !=0:
                return False
        return True
               
                
    
constantMonomial = [('',0,'')]

class constant(genpoly):
    def __init__(self,c):
        plist = [(c,constantMonomial)]
        genpoly.__init__(self,plist)
        self.polyType = 'genpoly'
        
class var(genpoly):
    def __init__(self,s):
        self.name = s
        genpoly.__init__(self,[(1,[(s,1,'pow')])])
        self.polyType = 'genpoly'
        
    def __str__(self,var=None,deg=None):
        return self.name    

    
def cos(p):
    if isinstance(p,numbers.Number):
        return constant(np.cos(p))
    
    if len(p.plist) == 1:
        c,mon = p.plist[0]

        if mon == constantMonomial:
            return constant(np.cos(c))
        
        if type(c) != int:
            raise TypeError("Coefficient must be an integer")
        
        var,deg,fun = mon[0]
        if (len(mon) > 1) or (deg > 1):
            raise ValueError('Variables cannot be multiplied')
            
        
        
        else:
            return genpoly([(1,[(var,abs(c),'cos')])])
        
    else:
        p0 = genpoly(p.plist[:1])
        p1 = genpoly(p.plist[1:])
        c0 = cos(p0)
        s0 = sin(p0)
        
        c1 = cos(p1)
        s1 = sin(p1)
        
        return c0*c1 - s0*s1
    
def sin(p):
    if isinstance(p,numbers.Number):
        return constant(np.sin(p))
    
    if len(p.plist) == 1:
        c,mon = p.plist[0]

        if mon == constantMonomial:
            return constant(np.sin(c))

        if type(c) != int:
            raise TypeError("Coefficient must be an integer")
        
        var,deg,fun = mon[0]
        if (len(mon) > 1) or (deg > 1):
            raise ValueError('Variables cannot be multiplied')
            
        
        else:
            return genpoly([(int(np.sign(c)),[(var,abs(c),'sin')])])
        
    else:
        p0 = genpoly(p.plist[:1])
        p1 = genpoly(p.plist[1:])
        c0 = cos(p0)
        s0 = sin(p0)
        
        c1 = cos(p1)
        s1 = sin(p1)
        
        return c0*s1 + s0*c1



def functify(p,v):
    if p is None:
        return None
    
    parr = ut.cast_as_array(p)
    pflat = parr.flatten()

    if isinstance(v,tuple):
        varrList = [ut.cast_as_array(vi) for vi in v]
        vflatList = [vi.flatten() for vi in varrList]
        vflat = np.hstack(vflatList)
    else:
        varr = ut.cast_as_array(v)
        vflat = varr.flatten()
    
    nP = len(pflat)
    nV = len(vflat)
    
    def fun(*x):
        
        xarrList = [ut.cast_as_array(xi) for xi in x]
        xflatList = [xi.flatten() for xi in xarrList]
        xflat = np.hstack(xflatList)
        valueDict = {vi.name : xi for vi,xi in zip(vflat,xflat)}
        finalval = np.zeros(nP)
        for i,pi in enumerate(pflat):
            finalval[i] = pi.val(valueDict)
        
        if len(parr.shape) > 0:
            finalval = finalval.reshape(parr.shape)
        else:
            finalval = finalval[0]
            
        return finalval
    
    return fun

def jacobian(p,x):
    parr = ut.cast_as_array(p)
    xarr = ut.cast_as_array(x)
    pflat = parr.flatten()
    xflat = xarr.flatten()

    nP = len(pflat)
    nX = len(xflat)
    Mat = np.zeros((nP,nX),dtype=object)
        
    for i,pi in enumerate(pflat):
        for j,xj in enumerate(xflat):
            Mat[i,j] = pi.diff(xj)
    return Mat.reshape(parr.shape+xarr.shape)

def subs(p,v,x):
    parr,varr,xarr = [ut.cast_as_array(z) for z in [p,v,x]]
    pflat,vflat,xflat = [z.flatten() for z in [parr,varr,xarr]]
    
    PSub = []
    for pi in pflat:
        for vj,xj in zip(vflat,xflat):
            pi = pi.sub(vj,xj)
        PSub.append(pi)
        
    if len(parr.shape) > 0:
        PSub = polyarr(PSub).reshape(parr.shape)
    else:
        PSub = PSub[0]
    return PSub

class polyarr(np.ndarray):
    def __new__(cls,inputarr):
        iarr = np.array(inputarr)
        iflat = iarr.flatten()
        pflat = np.zeros(iflat.shape,dtype=object)
        for i,xi in enumerate(iflat):
            if isinstance(xi,numbers.Number):
                pi = constant(xi)
            else:
                pi = xi
                
            pflat[i] = pi
        parr = pflat.reshape(iarr.shape)
        return np.asarray(parr).view(cls)
    
    def monomials(self):
        pflat = self.flatten()
        monFlat = np.zeros(len(pflat),dtype=object)
        for i,pi in enumerate(pflat):
            monFlat[i] = pi.monomials()
            
        return monFlat.reshape(self.shape)
    
    def allmonomials(self):
        mons = self.monomials().flatten()
        mons = reduce(lambda A,B : A+B,mons)
        mons.sort(key=lambda m : monomialString(m))
        
        newList = []
        mCur = mons[0]
        for m in mons[1:]:
            if m != mCur:
                newList.append(mCur)
                mCur = m
                
        newList.append(mCur)
        
        return newList

    def monomialArr(self):
       return polyarr([genpoly([(1,m)]) for m in self.allmonomials()])
   
    def variables(self):
        pflat = self.flatten()
        vFlat = np.zeros(len(pflat),dtype=object)
        for i,pi in enumerate(pflat):
            vFlat[i] = pi.variables()
            
        return vFlat.reshape(self.shape)
    
    def allvariables(self):
        Vars = self.variables().flatten()
        varList = reduce(lambda A,B : A+B,Vars)
        
        VarStrings = [v.name for v in varList]
        varList = list(set(VarStrings))
        varList.sort()
        return [var(v) for v in varList]

dot = ut.dot
outer = lambda u,v : polyarr(np.outer(u,v))
hstack = lambda lst : polyarr(np.hstack(lst)) 

def vararray(label,shape):
    NumEl = np.prod(shape)
    flatInd = range(NumEl)
    indices = np.unravel_index(flatInd,shape)
    flatArray = np.zeros(NumEl,dtype=object)
    for ind in flatInd:
        multiInd = []
        for arr in indices:
            multiInd.append(arr[ind])
            
        indexStrings = ['%d' % mi for mi in multiInd]
        indexString = '_'.join(indexStrings)
        varLabel = '%s_%s' % (label,indexString)
        newVar = var(varLabel)
        flatArray[ind] = newVar
    Arr = flatArray.reshape(shape)
    return polyarr(Arr)

def MonomialsUpTo(d,var_alg=[],var_trig=[]):
    if len(var_alg) > 0:
        MonList = []
        for k in range(d+1):
            MonList.append(var_alg[0]**k * MonomialsUpTo(d-k,var_alg[1:],var_trig))
        return hstack(MonList)

    elif len(var_trig) > 0:
        MonList = []
        for k in range(d+1):
            lowDegMons = MonomialsUpTo(d-k,[],var_trig[1:])
            MonList.append(cos(var_trig[0]*k) * lowDegMons)
            if k > 0:
                MonList.append(sin(var_trig[0]*k) * lowDegMons)
        return hstack(MonList)

    else:
        return polyarr([1.])

def pureMonomials(x,Arr):
    """
    Mons = pureMonomials(x,Arr)
    
    Return an array of monomials from Arr 
    whose variables are all contained in the vector x
    
    """
    Mons = Arr.monomialArr()
    xStrings = set([v.name for v in x.allvariables()])
    
    xMon = []
    
    for m in Mons:
        mVars = m.variables()
        mStrings = set([v.name for v in mVars])
        
        if mStrings.issubset(xStrings):
            xMon.append(m)
        
    return polyarr(xMon)

def proj(p,v):
    """
    C = proj(p,v)
    
    computes 
    
    argmin || Cv - p ||^2
    
    where the norm is given by standard inner product for polynomials. 
    
    Here 
    
    p - is a polynomial of class genpoly
    v - is a vector of class polyarr
    
    """
    Z = hstack([p,v])
    Mons = Z.allmonomials()
    At = np.zeros((len(v),len(Mons)))
    
    for i,vi in enumerate(v):
        for j,mj in enumerate(Mons):
            At[i,j] = vi.coeff(mj)
    A = At.T
    
    b = np.zeros(len(Mons))
    for j,mj in enumerate(Mons):
        b[j] = p.coeff(mj)
        
    y = la.lstsq(A,b)[0]
    return y

def residualMonomials(Arr,X):
    AllPolys = hstack([Arr.flatten(),X])
    AllMonoms = AllPolys.allmonomials()

    XMonoms = X.allmonomials()

    XMStringsSet = set([monomialString(mon) for mon in XMonoms])
    AllMStringSet = set([monomialString(mon) for mon in AllMonoms])
    
    UStringSet = AllMStringSet - XMStringsSet
    UMonoms = []
    for mon in AllMonoms:
        mStr = monomialString(mon)
        if mStr in UStringSet:
            UMonoms.append(mon)
        
    U = polyarr([genpoly([(1,mon)]) for mon in UMonoms])

    return U
