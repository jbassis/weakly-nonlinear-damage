"""
Calculate stream function, derivatives of stream function and stresses numerically for weakly non-linear problems
"""

import numpy as np
from numpy.linalg import solve
from scipy.interpolate import interp1d



rho_s = 0.0
rho_i = 910
rho_w = 1020
d_ice=rho_i/rho_w
xi=1/(1-d_ice)
zs = 0.5
zb =-0.5



def qmatrix(k,S0,ms=0.0,mb=0.0):
    """
    Calculate matrix terms such that Ws = qss*s+qsb*b and Wb = qbs*s + qbb*b
    Input: k (wavenumber), S0 stability number
    Output: qss,qsb,qbs,qbb
    """
    m = -(ms+mb)
    qss = (4*k**3*d_ice - 4*k**3 - 2*S0*np.sinh(2*k) - 4*k*S0)/(k*(-1 + d_ice)*(1 - np.cosh(2*k) + 2*k**2)) - m
    qsb = (4*np.sinh(k)*k**2*d_ice + 4*k*S0*np.cosh(k) + 4*np.sinh(k)*S0)/(k*d_ice*(1 - np.cosh(2*k) + 2*k**2))
    qbs = -(-4*np.sinh(k)*k**2*d_ice + 4*k*S0*np.cosh(k) + 4*np.sinh(k)*k**2 + 4*np.sinh(k)*S0)/(k*(-1 + d_ice)*(1 - np.cosh(2*k) + 2*k**2))
    qbb = (4*k**3*d_ice + 2*S0*np.sinh(2*k) + 4*k*S0)/(k*d_ice*(1 - np.cosh(2*k) + 2*k**2)) - m
    return qss,qsb,qbs,qbb

def eigs(k,S0,ms=0.0,mb=0.0):
    # Add melt rate to functions?
    qss,qsb,qbs,qbb=qmatrix(k,S0,ms=ms,mb=mb)
    b = -(qss+qbb)
    c = qss*qbb-qsb*qbs
    sigma1 = -b/2 + np.sqrt((b/2)**2-c)
    sigma2 = -b/2 - np.sqrt((b/2)**2-c)
    sigma = np.array([sigma1,sigma2])
    eig_vec = np.array([qsb,-qss])
    nrm = np.abs(eig_vec[0]-eig_vec[1])
    eig_vec = eig_vec/nrm
    return sigma,eig_vec

def Scrit(k,ms=0.0,mb=0.0):
    # Add melt rate to functions?
    #m = ms+mb
    #S = 2*(-4*k**2*np.exp(2*k) + np.exp(4*k) + np.sqrt((-d_ice**2*np.exp(4*k) 
    #                                                    + np.exp(4*k)*d_ice + 2*d_ice**2*np.exp(2*k)
    #                                                    - 2*np.exp(2*k)*d_ice - d_ice**2 + np.exp(2*k) 
    #                                                    + d_ice)*(4*k**2*np.exp(2*k) - np.exp(4*k) 
    #                                                        + 2*np.exp(2*k) - 1)**2*np.exp(-6*k))*np.exp(2*k) 
    #                                                        - 2*np.exp(2*k) + 1)*k**2*np.exp(2*k)/(-4*k**2*np.exp(6*k) 
    #                                                        + np.exp(8*k) + 8*k**2*np.exp(4*k) - 4*np.exp(6*k) 
    #                                                        - 4*k**2*np.exp(2*k) + 6*np.exp(4*k) - 4*np.exp(2*k) + 1)
    
    #print("S2",S,"no melt")
    
    # Need the total melt and need to convert from accumulation to melt
    m = -(ms+mb)
    S = -(((-4*k**2 - 4*k - 2)*m + 8*k)*np.exp(2*k) + ((16*k**3 + 8*k)*m - 32*k**3 
            + 16*np.sqrt(-((((m - 2)**2*k**2 + m**2)*d_ice**2 + (-(m - 2)**2*k**2 - m**2)*d_ice + k*m*(m - 2)/2)*np.exp(2*k) 
                + ((-2*(m - 2)**2*k**2 - (3*m**2)/2)*d_ice**2 + (2*(m - 2)**2*k**2 + (3*m**2)/2)*d_ice 
                   - (m - 2)**2*k**2 + m**2/8)*np.exp(4*k) + (((m - 2)**2*k**2 + m**2)*d_ice**2 + (-(m - 2)**2*k**2 - m**2)*d_ice 
                        - k*m*(m - 2)/2)*np.exp(6*k) - ((np.exp(8*k) + 1)*(-1/2 + d_ice)**2*m**2)/4)*((k**2 + 1/2)*np.exp(2*k) 
                            - np.exp(4*k)/4 - 1/4)**2*np.exp(-8*k)) - 16*k)*np.exp(4*k) 
                            + ((4*k**2 - 4*k + 2)*m + 8*k)*np.exp(6*k) - m*(np.exp(8*k) - 1))*k/((16*k**2 + 16)*np.exp(2*k) 
                                - 32*k**2*np.exp(4*k) + 16*k**2*np.exp(6*k) - 24*np.exp(4*k) + 16*np.exp(6*k) - 4*np.exp(8*k) - 4)

    
    return S


def psi(k,z):
    """
    Calculate stream function 
    Input: k: wavenumber k, 2*k, etc.
           z: vertical position 
    Returns np.array corresponding to coefficients of the 4 unknown coefficients 
    Example:
    >>psi(1,1/2)
    """
    return np.array([np.cosh(k*z),z*np.sinh(k*z),np.sinh(k*z),z*np.cosh(k*z)])

def diff_psi(k,z,p):
    """
    Calculate derivative of stream function 
    Input: k: wavenumber k, 2*k, etc.
           z: vertical position 
           p: order of derivative (1 = 1st derivative, 2 = 2nd derivative)
    Returns np.array corresponding to coefficients of the 4 unknown coefficients 
    Example:
    >>diff_psi(1,-1/2,2)
    """
    if np.mod(p,2)==0:
        t1=k**p*np.cosh(k*z)
        t2=z*k**p*np.sinh(k*z)+p*k**(p-1)*np.cosh(k*z)
        t3=k**p*np.sinh(k*z)
        t4=z*k**p*np.cosh(k*z)+p*k**(p-1)*np.sinh(k*z)
    else:
        t1=k**p*np.sinh(k*z)
        t2=z*k**p*np.cosh(k*z)+p*k**(p-1)*np.sinh(k*z)
        t3=k**p*np.cosh(k*z)
        t4=z*k**p*np.sinh(k*z)+p*k**(p-1)*np.cosh(k*z)
    return np.array([t1,t2,t3,t4])

def stress(k,z,deriv=0):
    """
    Calculate horizontal stress sigma_xx in weakly-nonlinear calculation
    Input k: wavenumber k, 2*k, etc.
          z: vertical position
          deriv: order of derivative
    """
    return -1/k*(diff_psi(k,z,3+deriv)+k**2*diff_psi(k,z,1+deriv))

def normal_stress(k,z):
    """
    Calculate normal stress for a harmonic mode in weakly-nonlinear calculation
    Input k: wavenumber k, 2*k, etc.
          z: vertical position
    """
    return 1/k*(-diff_psi(k,z,3)+3*k**2*diff_psi(k,z,1))

def shear_stress(k,z):
    """
    Calculate shear stress for a harmonic mode in weakly-nonlinear calculation
    Input k: wavenumber k, 2*k, etc.
          z: vertical position
    """
    return -(diff_psi(k,z,2)+k**2*psi(k,z))


def stress_bc(k,S0):
    """
    Calculate the homogeneous parts of the normal stress and shear stress lhs and rhs boundary conditions for 
    weakly-nonlinear expansion
    Input k: wavenumber 
          S0: Stability number
    Return A, rhs,
           A=4x4 matrix for coefficients of stream function
           rhs = homogenous part of forcing 
    """
    # Normal stress evaluated at zs
    eq1=normal_stress(k,zs)
    # Shear stress evaluated at zs
    eq2=shear_stress(k,zs)
    # Normal stress evaluated at zb
    eq3=normal_stress(k,zb)
    # Shear stress evaluated at zs
    eq4=shear_stress(k,zb)
    # Homogeneous rhs terms (first 2 are proportional to s, second proportional to b)
    rhs = np.array([[-4*S0*xi,-4*k, 4*S0/d_ice,-4*k]]).T 
    return np.vstack((eq1,eq2,eq3,eq4)),rhs



class StreamFun(object):
    def __init__(self,k,order,ms=0,mb=0):
        self.order = order
        self.k = k
        self.ms=ms
        self.mb=mb
        return
        
    def psi(self,z):
        k = self.order*self.k
        return  (np.dot(self.C.T,psi(k,z)))
    
    def stress(self,z,deriv):
        k = self.order*self.k
        return  (np.dot(self.C.T,stress(k,z,deriv)))
    
    def diff(self,z,deriv):
        k = self.order*self.k
        return  (np.dot(self.C.T,diff_psi(k,z,deriv)))
    
    def set_k(self,k):
        self.k =k 
        return None
        
    def __call__(self,z):
        return (np.dot(self.C.T,psi(self.k*self.order,z)))

    def solve(self,S0,rhs=None,kin=None):
        
        if self.order==1:
            A,r = stress_bc(self.k*self.order,S0)
            eig,vec = eigs(self.k,S0,self.ms,self.mb)
            r[0:2]=r[0:2]*vec[0]
            r[2:4]=r[2:4]*vec[1]
            self.C=solve(A,r)
            self.s=vec[0]
            self.b=vec[1]
            return self.C
        if self.order==0:
            self.C=np.array([0,0,0,0])
            h=rhs/(1+self.mb+self.ms) 
            self.h=h
            self.s=(1-d_ice)*h
            self.b=-d_ice*h
            return self.C,self.s,self.b
        else:
            A,r = stress_bc(self.k*self.order,S0)
            e1=np.vstack((-r[0:2],0,0))
            e2=np.vstack((0,0,-r[2:4]))
            A=np.hstack((A,e1,e2))
            # Switch to for melt rate??
            eq5 =np.hstack((-self.order*self.k*((psi(self.order*self.k,z=zs))),-self.ms-self.mb,0))
            eq6 =np.hstack((-self.order*self.k*((psi(self.order*self.k,z=zb))),0,-self.ms-self.mb))
            #eq5 =np.hstack((-self.order*self.k*((psi(self.order*self.k,z=zs))),0,0))
            #eq6 =np.hstack((-self.order*self.k*((psi(self.order*self.k,z=zb))),0,0))
            A = np.vstack((A,eq5,eq6))
            sol = solve(A,rhs)
            self.C=sol[0:4]
            self.s=sol[4]
            self.b=sol[5]
            return self.C,self.s,self.b
        return None
    
    def set_amp(self,s,b):
        self.s=s
        self.b=b
    

    def solveS(self,S0,rhs,funcs,S2=None,rhs_non_newt=np.array([0,0,0,0,0,0])):

        h11=funcs['h11']
        # Determine matrix components for stream function
        A,r = stress_bc(self.k,S0)


        # Terms proportional to S2
        S2_terms = np.array([[-4*h11.s*xi,
                               0,
                                4*h11.b/d_ice,
                               0]]).T 
        
        # stress rhs = (n-1)/n*diff(psi11)*A + terms * S4


        # We also get terms proportional to A from the Non-Newtonian terms?
        #non_newt_terms = ??

        # Solve for coefficients independent of S2, S4, etc, these will be proportional to the higher power of A
        C1=np.linalg.solve(A,rhs[0:4])


        # Solve for coefficients proportional to S2, S4, etc, these will be proportional to A
        C2=np.linalg.solve(A,S2_terms)

        #if rhs_non_newt != None:
            # Solve for coefficients proportional to non-newtonian terms??, these will be proportional to A
        C3 = np.linalg.solve(A,rhs_non_newt[0:4])
        # forcing terms unrelated to S2, S4, etc
        fs1=(np.dot(C1.T,self.k*psi(self.k,z=zs)))+rhs[4]
        fb1=(np.dot(C1.T,self.k*psi(self.k,z=zb)))+rhs[5]
  

        # forcing terms proportional to S2, S4, etc
        fs2=(np.dot(C2.T,self.k*psi(self.k,z=zs)))
        fb2=(np.dot(C2.T,self.k*psi(self.k,z=zb)))


        #if rhs_non_newt != None:
        # forcing terms proportional to non-Newtonian bits??
        fs3=(np.dot(C3.T,self.k*psi(self.k,z=zs))) + rhs_non_newt[4]
        fb3=(np.dot(C3.T,self.k*psi(self.k,z=zb))) + rhs_non_newt[5]
        #fb2 = fb2 + fb3 
        #fs2 = fs2 + fs3

        #print(fs1,fs2,fs2*S0,fs2*S2)
        #print(fb1,fb2,fb2*S0,fb2*S2)

        # The eigenvalues are singular so require special attention
        qss,qsb,qbs,qbb=qmatrix(self.k,S0)

        S2 = (qss*fb1-qbs*fs1)/(qbs*fs2-qss*fb2)

        self.C = np.linalg.solve(A,rhs[0:4]+S2_terms*S2)


        return S2[0]

    def set_amp_eqn(self,S0,S2=None,rhs=None,funcs=None,order=3):
        h11=funcs['h11']
    
            
        # Determine matrix components for stream function
        A,r = stress_bc(self.k,S0)

        # Terms proportional to S2
        S2_terms = np.array([[-4*h11.s*xi,
                                0,
                                4*h11.b/d_ice,
                                0]]).T 

        # Solve for coefficients independent of S2, S4, etc
        C1=np.linalg.solve(A,rhs[0:4])

        # Solve for coefficients proportional to S2, S4, etc
        C2=np.linalg.solve(A,S2_terms)

        # forcing terms unrelated to S2, S4, etc
        fs1=(np.dot(C1.T,self.k*psi(self.k,z=zs)))+rhs[4]
        fb1=(np.dot(C1.T,self.k*psi(self.k,z=zb)))+rhs[5]

        # forcing terms proportional to S2, S4, etc
        fs2=(np.dot(C2.T,self.k*psi(self.k,z=zs)))
        fb2=(np.dot(C2.T,self.k*psi(self.k,z=zb)))

        # The eigenvalues are singular so require special attention
        qss,qsb,qbs,qbb=qmatrix(self.k,S0)

        S2 = (qss*fb1-qbs*fs1)/(qbs*fs2-qss*fb2)

        self.C = np.linalg.solve(A,rhs[0:4]+S2_terms*S2)

        #qss*(fb1+fb2*S2)-qbs*(fs1+fs2*S2)=0
        #dAdt = (-qss*fb1+fb2*(S-S0)+qbs*fs2*(S-S0))/(qss*h11.s-qbs*h11.b)
        if order==3:
            t3 = -(-qss*fb1+qbs*fs1)/(qss*h11.b-qbs*h11.s)
            t1 = -(-qss*fb2+qbs*fs2)/(qss*h11.b-qbs*h11.s)
            self.t3=t3
            self.t1=t1
            def amp_eqn(S,S0,A):
                dAdt = self.t1*(S-S0)*A + self.t3*A**3
                return dAdt
        #elif order==5:
        #    t5 = -(-qss*fb1+qbs*fs1)/(qss*h11.b-qbs*h11.s)
        #    t3 = 
        #    t1 = -(-qss*fb2+qbs*fs2)/(qss*h11.b-qbs*h11.s)
        #    self.t3=t3
        #    self.t1=t1
        #    def amp_eqn(S,S0,S2,A):
        #        
        #        dAdt = self.t1*(S-S0-S2)*A + self.t3*A**3
        #        return dAdt

        self.amp_eqn = amp_eqn

class AmplitudeEquation(object):
    def __init__(self,S,h11,rhs31,rhs51,n=1):
        self.h11 = h11 
        self.rhs31 = rhs31
        self.rhs51 = rhs51
        self.S0=S[0]
        self.S2=S[1]
        self.S4=S[2]
        self.k = h11.k
        #self.t1_31,self.t3_31=self.set_amp_eqn(rhs31)
        #print(self.t1_31,-self.t1_31*self.S0,self.t3_31)
        self.t1,self.t5,self.t1_n=self.set_amp_eqn(rhs51,n)
        #print(self.t1,-self.t1*self.S0,-self.t1*self.S2,self.t5)
        return
    
    def set_amp_eqn(self,rhs,n):
        # Extract particular solution associated with lowest order mode
        h11 = self.h11
        part_sol = ParticularSol(h11)
        #print(part_sol.particular_sol(z=zs,n=3))
        #print("**n**",n)
        rhs_non_newt = part_sol.non_newtonian_rhs(n)
        #print(rhs_non_newt)
        # Determine matrix components for stream function
        A,r = stress_bc(self.k,self.S0)

        # Terms proportional to S2
        S2_terms = np.array([[-4*self.h11.s*xi,
                                0,
                                4*self.h11.b/d_ice,
                                0]]).T 

        # Solve for coefficients independent of S2, S4, etc
        C1=np.linalg.solve(A,rhs[0:4])
        

        # Solve for coefficients proportional to S2, S4, etc
        C2=np.linalg.solve(A,S2_terms)
        

        # Solve for coefficients proportional to non-Newtonian bits
        C3=np.linalg.solve(A,rhs_non_newt[0:4])
        #print(A)
        #print(rhs_non_newt[0:4])

        # forcing terms unrelated to S2, S4, etc
        fs1=(np.dot(C1.T,self.k*psi(self.k,z=zs)))+rhs[4]
        fb1=(np.dot(C1.T,self.k*psi(self.k,z=zb)))+rhs[5]

        # forcing terms proportional to S2, S4, etc
        fs2=(np.dot(C2.T,self.k*psi(self.k,z=zs)))
        fb2=(np.dot(C2.T,self.k*psi(self.k,z=zb)))

         # forcing terms proportional to S2, S4, etc
        fs3=(np.dot(C3.T,self.k*psi(self.k,z=zs)))+rhs_non_newt[4]
        fb3=(np.dot(C3.T,self.k*psi(self.k,z=zb)))+rhs_non_newt[5]

        # The eigenvalues are singular so require special attention
        qss,qsb,qbs,qbb=qmatrix(self.k,self.S0)

        t3 = -(-qss*fb1+qbs*fs1)/(qss*self.h11.b-qbs*self.h11.s)
        t1 = -(-qss*fb2+qbs*fs2)/(qss*self.h11.b-qbs*self.h11.s)

        t1_non_newt =  -(-qss*(fb3)+qbs*(fs3))/(qss*self.h11.b-qbs*self.h11.s)
        #print(-t1*(self.S0)+t1_non_newt)
        return t1, t3,t1_non_newt
    
    def amp_eqn_31(self,S,A):
        dAdt = self.t1*(S-self.S0)*A + self.t1_n*A + self.t5*A**3
        return dAdt
    
    def amp_eqn_51(self,S,A):
        dAdt = (self.t1*(S-self.S0) +self.t1_n)*A- self.t1*self.S2*A**3 + self.t5*A**5
    
    def __call__(self,S,A,order=5):
        if order == 3:
            return self.t1*(S-self.S0)*A + self.t5*A**3
        else:
            return self.t1*(S-self.S0)*A - self.t1*self.S2*A**3 + self.t5*A**5
        
    def Acrit(self,S,order=5):
        alpha = self.t1*(S-self.S0)+self.t1_n
        beta = - self.t1*self.S2
        gamma = self.t5
        A1 = np.zeros(np.shape(S))
        A2 = np.abs(np.sqrt(2)*np.sqrt(gamma*(-beta + np.sqrt(-4*alpha*gamma + beta**2)))/(2*gamma))
        A3 = np.abs(np.sqrt(-2*gamma*(beta + np.sqrt(-4*alpha*gamma + beta**2)))/(2*gamma))
        return [A1,A2,A3]
    


def amp2dam(k,amp,funcs):
    # Define position vector
    x = np.linspace(0,4*2*np.pi/k,101)
    h11=funcs['h11']
    h20=funcs['h20']
    h22=funcs['h22']
    h33=funcs['h33']
    h40=funcs['h40']
    h42=funcs['h42']
    dam = []
    for A in amp:
        surf = (1-d_ice) + (A*h11.s*np.cos(k*x)+(A**2*h22.s+A**4*h42.s)*np.cos(2*k*x)+A**3*h33.s*np.cos(3*k*x))
        #surf = ((A*(h11.s)*np.cos(k*x)+(A**2*(h22.s)+A**4*(h42.s))*np.cos(2*k*x)+A**3*(h33.s)*np.cos(3*k*x)))+(1-stream.d_ice)
        #bot =  ((A*(h11.b)*np.cos(k*x)+(A**2*(h22.b)+A**4*(h42.b))*np.cos(2*k*x)+A**3*(h33.b)*np.cos(3*k*x)))-stream.d_ice
        bot =  (-d_ice) + (A*h11.b*np.cos(k*x)+(A**2*h22.b+A**4*h42.b)*np.cos(2*k*x)+A**3*h33.b*np.cos(3*k*x))
        dam1 = 1-np.min(surf-bot)
        dam2 = np.max((A*(h11.s-h11.b)*np.cos(k*x)+(A**2*(h22.s-h22.b)+A**4*(h42.s-h42.b))*np.cos(2*k*x)+A**3*(h33.s-h33.b)*np.cos(3*k*x)))
        idx1 = np.argmax(surf-bot)
        idx2 = np.argmin(surf-bot)
        idx=np.argsort(surf-bot)
        h = surf-bot
        thick_min = h[idx[0]]
        thick_max1 = h[idx[-1]]
        thick_max2 = h[idx[-2]]
        #thick_mean = 0.5*(thick_max1+thick_max2)
        thick_mean=np.mean(h)+(A**2*(h20.s[0]-h20.b[0])+A**4*(h40.s[0]-h40.b[0]))
        #id_maxes=argrelextrema(surf-bot, np.less)
        
        #thick1=surf[id_maxes[0]]-bot[id_maxes[0]]
        #thick2=surf[id_maxes[1]]-bot[id_maxes[1]]
        #mean_thick = 0.5*(thick_max1+thick_max2)
        #dam3=1-(surf[idx2]-bot[idx2])/((surf[idx1]-bot[idx1]))
        dam3=1-thick_min/thick_mean
        #dam3 = 1-((np.min(surf)-np.max(bot)))
        dam.append(dam3)
        #dam.append(1-np.min(surf-bot))
        #dam.append(np.max((A*(h11.s-h11.b)*np.cos(k*x)+(A**2*(h22.s-h22.b)+A**4*(h42.s-h42.b))*np.cos(2*k*x)+A**3*(h33.s-h33.b)*np.cos(3*k*x))))
    return dam

def dam2amp(k,damage,funcs):
    """
    Convert damage to amplitude
    Note that we do this by interpolating after finding a relationship between
    linearly spaced amplitudes and damage
    """
    h11=funcs['h11']
    h20=funcs['h20']
    h22=funcs['h22']
    h33=funcs['h33']
    h40=funcs['h40']
    h42=funcs['h42']
    A = np.linspace(0,1.0,50001)
    dam = np.array(amp2dam(k,A,funcs))
    dam_fun = interp1d(dam,A,kind='linear')
    amp = dam_fun(damage)
    return amp

def topo(k,A,funcs):
    h11=funcs['h11']
    h20=funcs['h20']
    h22=funcs['h22']
    h33=funcs['h33']
    h40=funcs['h40']
    h42=funcs['h42']

    x = np.linspace(0,8*2*np.pi/k,501)

    surf = (1-d_ice) + (A*h11.s*np.cos(k*x)+(A**2*h22.s + A**4*h42.s)*np.cos(2*k*x)+A**3*h33.s*np.cos(3*k*x))+A**2*h20.s+0*A**4*h40.s 
    bot =  (-d_ice)  + (A*h11.b*np.cos(k*x)+(A**2*h22.b + A**4*h42.b)*np.cos(2*k*x)+A**3*h33.b*np.cos(3*k*x))+A**2*h20.b+0*A**4*h40.b

    return x,surf,bot


class ParticularSol(object):
    def __init__(self,psi11):
        self.C = psi11.C
        self.psi11 = psi11
        self.k = psi11.k
        return
        
    def particular_sol(self,z,n):
        C = self.C
        k = self.k
        sol = (-k**2*C[1]*(n - 1)*z**3/(6*n) - k*(n - 1)*(k*C[2] + C[3])*z**2/(2*n))*np.sinh(k*z) + (-k**2*C[3]*(n - 1)*z**3/(6*n) - k*(n - 1)*(k*C[0] + C[1])*z**2/(2*n))*np.cosh(k*z)
        return  sol
    
    def normal_stress(self,k,z,n):
        C= self.C
        diff_psi11_1 = -k*((z*(z*C[1] + 3*C[2])*k**2 + (6*z*C[3] + 6*C[0])*k + 6*C[1])*np.cosh(k*z) 
                       + (z*(z*C[3] + 3*C[0])*k**2 + (6*z*C[1] + 6*C[2])*k + 6*C[3])*np.sinh(k*z))*z*(n - 1)/(6*n)  
        diff_psi11_3 =  -k**2*(n - 1)*((z**2*(z*C[1] + 3*C[2])*k**3 + (12*z**2*C[3] + 18*z*C[0])*k**2 + (36*z*C[1] + 18*C[2])*k + 24*C[3])*np.cosh(k*z) 
                                  + (z**2*(z*C[3] + 3*C[0])*k**3 + (12*z**2*C[1] + 18*z*C[2])*k**2 + (36*z*C[3] + 18*C[0])*k + 24*C[1])*np.sinh(k*z))/(6*n)
        normal_stress =1/k*(-diff_psi11_3+3*k**2*diff_psi11_1)
        return normal_stress
    
    def shear_stress(self,k,z,n):
        C=self.C
        diff_psi_2 =  -(((z**2*(z*C[3] + 3*C[0])*k**3 + (9*z**2*C[1] + 12*z*C[2])*k**2 
                          + (18*z*C[3] + 6*C[0])*k + 6*C[1])*np.cosh(k*z) 
                          +  np.sinh(k*z)*(z**2*(z*C[1] + 3*C[2])*k**3 + (9*z**2*C[3] + 12*z*C[0])*k**2 
                                           + (18*z*C[1] + 6*C[2])*k + 6*C[3]))*(n - 1)*k)/(6*n)
    
        shear = -(diff_psi_2 + k**2*self.particular_sol(z,n))
        return shear
    
    def stress_rhs(self,n):
        psi11=self.psi11
        norm_stress_top = -self.normal_stress(self.k,z=zs,n=n)-4*(1/n - 1)*self.k*psi11.diff(z=zs,deriv=1)
        norm_stress_bot = -self.normal_stress(self.k,z=zb,n=n)-4*(1/n - 1)*self.k*psi11.diff(z=zb,deriv=1)
        shear_stress_top = -self.shear_stress(self.k,z=zs,n=n)
        shear_stress_bot = -self.shear_stress(self.k,z=zb,n=n)
        #return np.array([-4*(1/n - 1)*self.k*psi11.diff(z=zs,deriv=1)[0],0,-4*(1/n - 1)*self.k*psi11.diff(z=zb,deriv=1)[0],0]).reshape((4,1))
        return np.array([norm_stress_top[0],shear_stress_top[0],norm_stress_bot[0],shear_stress_bot[0]]).reshape((4,1))
    
    def kinematic_rhs(self,n):
        return np.array([self.k*self.particular_sol(zs,n)[0],self.k*self.particular_sol(zb,n)[0]]).reshape((2,1))
    
    def non_newtonian_rhs(self,n):
        stress_bc = self.stress_rhs(n)
        kin_bc =self.kinematic_rhs(n)
        rhs_forcing_non_newtonian = np.vstack((stress_bc,kin_bc))
        #print(rhs_forcing_non_newtonian)
        return rhs_forcing_non_newtonian

    







            




    


        
    
    

    



    






   
    
   

        





       
   


     



    
    

