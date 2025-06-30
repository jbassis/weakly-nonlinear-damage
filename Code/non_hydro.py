"""
"""

import numpy as np
from numpy.linalg import solve
import stress_stream as stream

d_ice=stream.d_ice#910/1020
xi=stream.xi#1/(1-d_ice)
zs = stream.zs#0.5
zb =stream.zb#-0.5

def rhs22(k,S0,funcs):
    """
   
    """
    Ss = -4*S0*xi
    Sb = 4*S0/d_ice
    h11=funcs["h11"]
    # Part not proportional to amplitude
    s11 = h11.s
    b11 = h11.b
    tau_xx_11s=h11.stress(zs,deriv=0)[0]
    tau_xx_11b=h11.stress(zb,deriv=0)[0]


    # Rhs forcing terms related to lower-order forcing
    rhs_forcing = np.array([[ - 4*k**2*s11**2, 
                                - s11*k*tau_xx_11s + k*s11**2*Ss/2, 
                                - 4*k**2*b11**2, 
                                - b11*k*tau_xx_11b + k*b11**2*Sb/2,
                                +(0.5*h11.diff(z=zs,deriv=1)*h11.s*k)[0],
                                +(0.5*h11.diff(z=zb,deriv=1)*b11*k)[0]]]).T 
    return rhs_forcing


def rhs20(k,S0,funcs):
    """
   
    """
    h11=funcs['h11']

    # Part not proportional to amplitude
    s11 = h11.s
    b11 = h11.b
    
    # Rhs forcing terms for kinematic equation related to lower-order terms
    rhs_forcing=(0.5*h11.diff(z=zs,deriv=1)*s11*k) -(0.5*h11.diff(z=zb,deriv=1)*b11*k) 
    
    
    return rhs_forcing


def rhs33(k,S0,funcs):
    """
   
    """
    h11=funcs["h11"]
    h22=funcs["h22"]

    # Part not proportional to amplitude
    s11 = h11.s
    b11 = h11.b

    s22 = h22.s
    b22 = h22.b


    tau_xx_11s=h11.stress(zs,deriv=0)[0]
    tau_xx_11b=h11.stress(zb,deriv=0)[0]

    tau_xx_11zs = h11.stress(zs,deriv=1)[0]
    tau_xx_11zb = h11.stress(zb,deriv=1)[0]

    tau_xx_22s=h22.stress(zs,deriv=0)[0]
    tau_xx_22b=h22.stress(zb,deriv=0)[0]


    # Stress boundary conditions

    # Rhs forcing terms related to lower-order forcing
    stress_bc_rhs = np.array([ -4*(3*k**2*s11**3*xi*S0)/4 - (9*tau_xx_11s*k**2*s11**2)/8 - 18*k**2*s11*s22, 
                                -(3*k*(tau_xx_11zs*s11**2 + 4*4*s11*xi*S0*s22 + 4*tau_xx_11s*s22 + 4*tau_xx_22s*s11))/8 + s11**3*k**2/3, 
                                4*(3*k**2*b11**3*S0/d_ice)/4 - (9*tau_xx_11b*k**2*b11**2)/8 - 18*k**2*b11*b22, 
                                -(3*k*(tau_xx_11zb*b11**2 - 4*4*b11*S0*b22/d_ice + 4*tau_xx_11b*b22 + 4*tau_xx_22b*b11))/8 + b11**3*k**2/3]) 

    # Rhs forcing terms for kinematic equation related to lower-order terms
    kin_ode_zs = ((0.5*h11.diff(z=zs,deriv=1)*s22*k) + h22.diff(z=zs,deriv=1)*s11*k +h11.diff(z=zs,deriv=2)*s11**2*k/8)
    kin_ode_zb = ((0.5*h11.diff(z=zb,deriv=1)*b22*k) + h22.diff(z=zb,deriv=1)*b11*k +h11.diff(z=zb,deriv=2)*b11**2*k/8)    

    rhs_forcing = np.vstack((stress_bc_rhs,kin_ode_zs,kin_ode_zb))
    #print(rhs_forcing)

    return rhs_forcing

def rhs31(k,S0,funcs,S=None):
    """
    Third order amplitude equation
    Two options:
    1. Calculate S2 for steady-state solutions
    2. Calculate dA/dt
    """
    h11 = funcs['h11']
    h22 = funcs['h22']
    h20 = funcs['h20']
    if S!=None:
        S2 = S-S0
        #print("S2",S2)
    else:
        S2 =0

    #Ss = -4*S0/(1-d_ice)
    #Sb =  4*S0/d_ice
    # Part not proportional to amplitude
    s11 = h11.s
    b11 = h11.b
   

    s22 = h22.s
    b22 = h22.b

    s20 = h20.s
    b20 = h20.b


    tau_xx_11s=h11.stress(zs,deriv=0)[0]
    tau_xx_11b=h11.stress(zb,deriv=0)[0]

    tau_xx_11zs = h11.stress(zs,deriv=1)[0]
    tau_xx_11zb = h11.stress(zb,deriv=1)[0]

    tau_xx_22s=h22.stress(zs,deriv=0)[0]
    tau_xx_22b=h22.stress(zb,deriv=0)[0]

    tau_xx_20=h20.stress


    # Rhs forcing terms related to lower-order forcing
    stress_bc_rhs = np.array([-4*k**2*s11**3*xi*S0/4 - (3*tau_xx_11s*k**2*s11**2)/8 - 4*k**2*s11*s20 - 2*k**2*s11*s22,
                               -k*((3*tau_xx_11zs*s11**2)/8 + (s20 + s22/2)*tau_xx_11s + tau_xx_22s*s11/2 + (4*s20*xi*S0 + 4*1/2*s22*xi*S0 + tau_xx_20)*s11) + s11**3*k**2,
                                4*k**2*b11**3*S0/d_ice/4 - (3*tau_xx_11b*k**2*b11**2)/8 - 4*k**2*b11*b20 - 2*k**2*b11*b22,
                               -k*((3*tau_xx_11zb*b11**2)/8 + (b20 + b22/2)*tau_xx_11b + tau_xx_22b*b11/2 + (-4*b20*S0/d_ice - 4*1/2*b22*S0/d_ice + tau_xx_20)*b11) + b11**3*k**2])
    
    stress_bc_rhs_S2 = np.array([[-4*h11.s*xi*S2,
                               0,
                                4*h11.b/d_ice*S2,
                               0]]).T 
    
    # Rhs forcing terms for kinematic equation related to lower-order terms
    kin_ode_zs = ((3*h11.diff(z=zs,deriv=2)*k*s11**2)/8 + (s20 + s22/2)*k*h11.diff(z=zs,deriv=1) + h22.diff(z=zs,deriv=1)*k*s11)[0]
    kin_ode_zb = ((3*h11.diff(z=zb,deriv=2)*k*b11**2)/8 + (b20 + b22/2)*k*h11.diff(z=zb,deriv=1) + h22.diff(z=zb,deriv=1)*k*b11)[0]

    # This is for a solution for S2?
    rhs_forcing = np.vstack((stress_bc_rhs,kin_ode_zs,kin_ode_zb))
    return rhs_forcing
    """
    
    # The 31 term is fundamentally different because h31=0 and we need to solve for S2
    if S2 == 0:
        rhs = np.array([[4*h11.h,0,4*h11.h,0]]).T
        A = np.hstack((A,rhs))
         # Kinematic bc terms
        eq5 = -k*((psi(k,z=zs))-(psi(k,z=zb)))
        eq5 = (np.append(eq5,0))
        B=np.vstack((A,eq5))
        rhs = np.vstack((stress_bc_rhs,kin_bc_rhs))

        sol=solve(B,rhs)
        C31 = sol[0:4]
        S2 = sol[-1][0]
        return C31,S2
    else:
        # Part of C31 without S2 (proportional to A^3)
        C31=solve(A,stress_bc_rhs)
        # Part of C31 with S2 (propotional to A)
        C31_S2 = solve(A,stress_bc_rhs_S2)
        kin_bc = k*((psi(k,z=zs))-(psi(k,z=zb)))
        # Terms in dAdt proportional to A^3
        dAdt_3=(np.dot(C31.T,kin_bc)+kin_bc_rhs)[0]
        # Terms in dAdt proportional to A
        dAdt_1=(np.dot(C31_S2.T,kin_bc))[0]
        #print("dAdt",dAdt_1,"A","+",dAdt_3,"A^3")
        return [dAdt_1,dAdt_3]
    """
    




def rhs40(k,S,funcs):
    """
   
    """
    S0 = S[0]
    S2 = S[1]
    h11=funcs['h11']
    h22=funcs['h22']
    h20=funcs['h20']
    h33=funcs['h33']
    h31=funcs['h31']
    # Part not proportional to amplitude
    s11 = h11.s
    b11 = h11.b

    s22 = h22.s
    b22 = h22.b

    s20 = h20.s
    b20 = h20.b 
   
    s31 = h31.s
    b31 = h31.b


    C=np.array([0,0,0,0])


    # Rhs forcing terms for kinematic equation related to lower-order terms
    kin_odes = h11.diff(z=zs,deriv=3)*k*s11**3/16 \
             + ((s20 + s22/2)*s11*k*h11.diff(z=zs,deriv=2))/2 \
             + h22.diff(z=zs,deriv=2)*k*s11**2/4  \
             + h22.diff(z=zs,deriv=1)*s22*k \
             + h31.diff(z=zs,deriv=1)*s11*k/2

    kin_odeb = h11.diff(z=zb,deriv=3)*k*b11**3/16  \
             + ((b20 + b22/2)*b11*k*h11.diff(z=zb,deriv=2))/2 \
             + h22.diff(z=zb,deriv=2)*k*b11**2/4 \
             + h22.diff(z=zb,deriv=1)*b22*k \
             + h31.diff(z=zb,deriv=1)*b11*k/2


    rhs_forcing = (kin_odes-kin_odeb)
    


    
    #tau_xx_40 = -4*S0*h40-4*S2*h20.h
    
    return rhs_forcing
            
def rhs42(k,S,funcs):
    """
   
    """
    S0=S[0]
    S2=S[1]
    h11=funcs['h11']
    h22=funcs['h22']
    h31=funcs['h31']
    h33=funcs['h33']
    h20=funcs['h20']

    # Part not proportional to amplitude
    s11 = h11.s
    b11 = h11.b

    s20 = h20.s
    b20 = h20.b

    s22 = h22.s
    b22 = h22.b

    s33 = h33.s
    b33 = h33.b


    tau_xx_11s=h11.stress(zs,deriv=0)[0]
    tau_xx_11b=h11.stress(zb,deriv=0)[0]

    tau_xx_11zs = h11.stress(zs,deriv=1)[0]
    tau_xx_11zb = h11.stress(zb,deriv=1)[0]

    tau_xx_11zzs = h11.stress(zs,deriv=2)[0]
    tau_xx_11zzb = h11.stress(zb,deriv=2)[0]

    tau_xx_22s=h22.stress(zs,deriv=0)[0]
    tau_xx_22b=h22.stress(zb,deriv=0)[0]

    tau_xx_22zs = h22.stress(zs,deriv=1)[0]
    tau_xx_22zb = h22.stress(zb,deriv=1)[0]

    tau_xx_31s=h31.stress(zs,deriv=0)[0]
    tau_xx_31b=h31.stress(zb,deriv=0)[0]

    tau_xx_33s=h33.stress(zs,deriv=0)[0]
    tau_xx_33b=h33.stress(zb,deriv=0)[0]

    tau_xx_20=h20.stress


    # Rhs forcing terms related to lower-order forcing
    stress_bc_rhs = np.array([ -(2*tau_xx_11zs*k**2*s11**3)/3 - 2*k**2*s11*(s20 + s22)*tau_xx_11s - tau_xx_22s*k**2*s11**2 + ((-4*2*s20*xi*S0 - 4*2*s22*xi*S0 - tau_xx_20)*s11**2 - 8*(s33)*s11 - 16*s20*s22)*k**2 - 4*xi*s22*S2 - (2*s11**4*k**4)/3,
                               ((- s33)*tau_xx_11s - s11**3*tau_xx_11zzs/6 - s11*(s20 + s22)*tau_xx_11zs - tau_xx_22zs*s11**2/2 - 2*tau_xx_22s*s20 - tau_xx_31s*s11 - tau_xx_33s*s11 - 4*s11**2*xi*S2/2 - 4*xi*S0*(s33)*s11 - 4*2*s22*s20*xi*S0 - 2*s22*tau_xx_20)*k +(-4*xi*s11**4*S0*k**3/12 - 4*s11**2*k**3*s22),
                               -(2*tau_xx_11zb*k**2*b11**3)/3 - 2*k**2*b11*(b20 + b22)*tau_xx_11b - tau_xx_22b*k**2*b11**2 + ((4*2*b20*S0/d_ice + 4*2*b22*S0/d_ice - tau_xx_20)*b11**2 - 8*(b33)*b11 - 16*b20*b22)*k**2 + 4*b22*S2/d_ice - (2*b11**4*k**4)/3,
                               ((- b33)*tau_xx_11b - b11**3*tau_xx_11zzb/6 - b11*(b20 + b22)*tau_xx_11zb - tau_xx_22zb*b11**2/2 - 2*tau_xx_22b*b20 - tau_xx_31b*b11 - tau_xx_33b*b11 + 4*b11**2*S2/2/d_ice + 4*S0*(b33)*b11/d_ice + 4*2*b22*b20*S0/d_ice - 2*b22*tau_xx_20)*k + (4*b11**4*S0*k**3/12/d_ice - 4*b11**2*k**3*b22)]) 
    
  

    # Rhs forcing terms for kinematic equation related to lower-order terms
    kin_ode_zs = -0.5*(-s22*s11*h11.diff(z=zs,deriv=2)*k - s20*s11*h11.diff(z=zs,deriv=2)*k - h11.diff(z=zs,deriv=1)*s33*k \
                 - h31.diff(z=zs,deriv=1)*s11*k - 3*h33.diff(z=zs,deriv=1)*s11*k - h22.diff(z=zs,deriv=2)*k*s11**2 \
                 - h11.diff(z=zs,deriv=3)*k*s11**3/6  - 4*h22.diff(z=zs,deriv=1)*s20*k)[0] 
                 
    kin_ode_zb = -0.5*(-b22*b11*h11.diff(z=zb,deriv=2)*k - b20*b11*h11.diff(z=zb,deriv=2)*k - h11.diff(z=zb,deriv=1)*b33*k \
                 -h31.diff(z=zb,deriv=1)*b11*k - 3*h33.diff(z=zb,deriv=1)*b11*k - h22.diff(z=zb,deriv=2)*k*b11**2 \
                 - h11.diff(z=zb,deriv=3)*k*b11**3/6  - 4*h22.diff(z=zb,deriv=1)*b20*k)[0] 
                

   

    rhs_forcing = np.vstack((stress_bc_rhs,kin_ode_zs,kin_ode_zb))

   
    return rhs_forcing


    
def rhs51(k,S,funcs):
    """
    Fifth order amplitude equation
    Two options:
    1. Calculate S4 for steady-state solutions
    2. Calculate dA/dt
    """
    S0=S[0]
    S2=S[1]
    h11=funcs['h11']
    h22=funcs['h22']
    h20=funcs['h20']
    h31=funcs['h31']
    h33=funcs['h33']
    h40=funcs['h40']
    h42=funcs['h42']


    S0s= -4*S0*xi
    S0b=  4*S0/d_ice 
    S2s = -4*S2*xi
    S2b = 4*S2/d_ice
    # Part not proportional to amplitude
    s11 = h11.s
    b11 = h11.b
   

    s22 = h22.s
    b22 = h22.b

    s20 = h20.s
    b20 = h20.b

    s31 = h31.s
    b31 = h31.b

    s33 = h33.s
    b33 = h33.b

    s40 = h40.s
    b40 = h40.b

    s42 = h42.s
    b42 = h42.b


    tau_xx_11s=h11.stress(zs,deriv=0)[0]
    tau_xx_11b=h11.stress(zb,deriv=0)[0]

    tau_xx_11zs = h11.stress(zs,deriv=1)[0]
    tau_xx_11zb = h11.stress(zb,deriv=1)[0]

    tau_xx_11zzs = h11.stress(zs,deriv=2)[0]
    tau_xx_11zzb = h11.stress(zb,deriv=2)[0]

    tau_xx_11zzzs = h11.stress(zs,deriv=3)[0]
    tau_xx_11zzzb = h11.stress(zb,deriv=3)[0]

    tau_xx_22s=h22.stress(zs,deriv=0)[0]
    tau_xx_22b=h22.stress(zb,deriv=0)[0]

    tau_xx_22zs=h22.stress(zs,deriv=1)[0]
    tau_xx_22zb=h22.stress(zb,deriv=1)[0]

    tau_xx_22zzs=h22.stress(zs,deriv=2)[0]
    tau_xx_22zzb=h22.stress(zb,deriv=2)[0]

    tau_xx_31s=h31.stress(zs,deriv=0)[0]
    tau_xx_31b=h31.stress(zb,deriv=0)[0]

    tau_xx_31zs=h31.stress(zs,deriv=1)[0]
    tau_xx_31zb=h31.stress(zb,deriv=1)[0]

    tau_xx_33s=h33.stress(zs,deriv=0)[0]
    tau_xx_33b=h33.stress(zb,deriv=0)[0]

    tau_xx_33zs=h33.stress(zs,deriv=1)[0]
    tau_xx_33zb=h33.stress(zb,deriv=1)[0]

    tau_xx_42s=h42.stress(zs,deriv=0)[0]
    tau_xx_42b=h42.stress(zb,deriv=0)[0]

    tau_xx_20=h20.stress

    tau_xx_40=h40.stress


    fns = -(5*tau_xx_11zzs*k**2*s11**4)/64 - (3*k**2*(s20 + (2*s22)/3)*s11**2*tau_xx_11zs)/4 - k**2*tau_xx_22zs*s11**3/6 \
          - k**2*((s33)*s11 + 2*s20**2 + 2*s20*s22 + s22**2)*tau_xx_11s/4 - k**2*s11*(s20 + s22)*tau_xx_22s/2 \
          - (3*tau_xx_31s*k**2*s11**2)/8 - tau_xx_33s*k**2*s11**2/8 \
          + ((s11**3*S2s + S0s*(s33)*s11**2 + 2*(s22**2*S0s + (2*s20*S0s - tau_xx_20)*s22 + 2*S0s*s20**2 - 2*tau_xx_20*s20 - 8*(s40 + s42/2))*s11 - 8*((s33)*s22 + 2*0*s20))*k**2)/4  \
          - (s20 - (3*s22)/2)*s11**3*k**4 + (-k**4*tau_xx_11s*s11**4/24 + s11**5*k**4*S0s/24 - (17*s11**3*k**4*s22)/6)
    
    
    fnb = -(5*tau_xx_11zzb*k**2*b11**4)/64 - (3*k**2*(b20 + (2*b22)/3)*b11**2*tau_xx_11zb)/4 - k**2*tau_xx_22zb*b11**3/6 \
          - k**2*((b33)*b11 + 2*b20**2 + 2*b20*b22 + b22**2)*tau_xx_11b/4 - k**2*b11*(b20 + b22)*tau_xx_22b/2 \
          - (3*tau_xx_31b*k**2*b11**2)/8 - tau_xx_33b*k**2*b11**2/8 \
          + ((4*b11**3*(S2/d_ice) + 4*(S0/d_ice)*b33*b11**2 + 2*(-4*xi*b22**2*(-S0/xi/d_ice) + (-(4*xi)*2*b20*(-S0/xi/d_ice) - tau_xx_20)*b22 - (4*xi)*2*(-S0/xi/d_ice)*b20**2 - 2*tau_xx_20*b20 - 8*(b40 + b42/2))*b11 - 8*((b33)*b22 ))*k**2)/4 \
          - (b20 - (3*b22)/2)*b11**3*k**4 + (-k**4*tau_xx_11b*b11**4/24 - 4*xi*b11**5*k**4*(-S0/xi/d_ice)/24 - (17*b11**3*k**4*b22)/6)
    

    fss = -k*((5*s11**4*tau_xx_11zzzs)/96 + (3*s11**2*(s20 + (2*s22)/3)*tau_xx_11zzs)/4 
              + tau_xx_22zzs*s11**3/6 + (-(3*s11**4*k**2)/8 + ((s33)*s11)/2 + s20*s22 + s20**2 
                                         + s22**2/2)*tau_xx_11zs + s11*(s20 + s22)*tau_xx_22zs 
                                         + (3*s11**2*tau_xx_31zs)/4 + s11**2*tau_xx_33zs/4 
                                         + (-k**2*(s22/2 + s20)*s11**2 + 2*s40 + s42)*tau_xx_11s + (-s11**3*k**2/2  + s33)*tau_xx_22s 
                                         + (2*s20 + s22)*tau_xx_31s + tau_xx_33s*s22 
                                         + s11*tau_xx_42s + k**2*(-tau_xx_20 + 1/2*s22*(-4*xi*S0) + s20*(-4*xi*S0))*s11**3 
                                         + (-2*s20*(-4*xi*S2) - s22*(-4*xi*S2) 
                                         + (-2*s40 - s42)*(-4*xi*S0) + 2*tau_xx_40)*s11 - (-4*xi*S0)*(s33)*s22 )/2 + (-tau_xx_11s*k**3*s11**2*s22/8 
                                                                                                                      + (3*tau_xx_11s*k**3*s11**2*s20)/4 
                                                                                                                      - s11**3*k**3*s20*(-4*xi*S0)/2 
                                                                                                                      - s11**3*k**3*s22*(-4*xi*S0)/24 
                                                                                                                      + 3*k**3*1*s11**2*s33 
                                                                                                                      - 8*s11*s22**2*k**3*1 
                                                                                                                      + s11**5*k**5/12 
                                                                                                                      + s11**3*k**3*tau_xx_20/2 
                                                                                                                      + (23*tau_xx_11zs*k**3*s11**4)/96 
                                                                                                                      + (13*tau_xx_22s*s11**3*k**3)/24)
    
    
    fsb = -k*((5*b11**4*tau_xx_11zzzb)/96 + (3*b11**2*(b20 + (2*b22)/3)*tau_xx_11zzb)/4 
              + tau_xx_22zzb*b11**3/6 + (-(3*b11**4*k**2)/8 + ((b33)*b11)/2 + b20*b22 + b20**2 + b22**2/2)*tau_xx_11zb 
              + b11*(b20 + b22)*tau_xx_22zb + (3*b11**2*tau_xx_31zb)/4 + b11**2*tau_xx_33zb/4 
              + (-k**2*(b22/2 + b20)*b11**2 + 2*b40 + b42)*tau_xx_11b + (-b11**3*k**2/2+ b33)*tau_xx_22b 
              + (2*b20 + b22)*tau_xx_31b + tau_xx_33b*b22 + b11*tau_xx_42b + k**2*(-tau_xx_20 + 1/2*b22*S0b + b20*S0b)*b11**3  
              + (-2*S2b*b20 - b22*S2b + (-2*b40 - b42)*S0b + 2*tau_xx_40)*b11 
              - S0b*(b33)*b22)/2 + (-tau_xx_11b*k**3*b11**2*b22/8 + (3*tau_xx_11b*k**3*b11**2*b20)/4 - b11**3*k**3*b20*S0b/2 
                                    - b11**3*k**3*b22*S0b/24 + 3*k**3*b11**2*b33 - 8*b11*b22**2*k**3 + b11**5*k**5/12 
                                    + b11**3*k**3*tau_xx_20/2 + (23*tau_xx_11zb*k**3*b11**4)/96 + (13*tau_xx_22b*b11**3*k**3)/24)
   

    stress_bc_rhs = np.array([fns,fss,fnb,fsb])


    
    #print(stress_bc_rhs)
    kin_ode_zs = -5*(s11**2 + (72*s20)/5 + (48*s22)/5)*s11**2*k*h11.diff(z=zs,deriv=3)/192 - h22.diff(z=zs,deriv=3)*k*s11**3/6 \
                 - 3*((s31 + s33/3)*s11 + (2*s20**2)/3 + (2*s22*s20)/3 + s22**2/3)*k*h11.diff(z=zs,deriv=2)/4 \
                 - k*s11*(s20 + s22)*h22.diff(z=zs,deriv=2) - (3*s11**2*h31.diff(z=zs,deriv=2)*k)/8 \
                 - (3*s11**2*h33.diff(z=zs,deriv=2)*k)/8 - (s42/2 + s40)*k*h11.diff(z=zs,deriv=1) \
                 - k*(s33)*h22.diff(z=zs,deriv=1) - (s20 + s22/2)*k*h31.diff(z=zs,deriv=1) \
                 - (3*h33.diff(z=zs,deriv=1)*k*s22)/2 - h11.psi(z=zs)*h42.diff(z=zs,deriv=1)*k #- psi51*k
    
    kin_ode_zb =-5*(b11**2 + (72*b20)/5 + (48*b22)/5)*b11**2*k*h11.diff(z=zb,deriv=3)/192 - h22.diff(z=zb,deriv=3)*k*b11**3/6 \
                - 3*((b31 + b33/3)*b11 + (2*b20**2)/3 + (2*b22*b20)/3 + b22**2/3)*k*h11.diff(z=zb,deriv=2)/4 \
                - k*b11*(b20 + b22)*h22.diff(z=zb,deriv=2) - (3*b11**2*h31.diff(z=zb,deriv=2)*k)/8 \
                - (3*b11**2*h33.diff(z=zb,deriv=2)*k)/8 - (b42/2 + b40)*k*h11.diff(z=zb,deriv=1) \
                - k*(b31 + b33)*h22.diff(z=zb,deriv=1) - (b20 + b22/2)*k*h31.diff(z=zb,deriv=1) \
                - (3*h33.diff(z=zb,deriv=1)*k*b22)/2 - h11.psi(z=zb)*h42.diff(z=zb,deriv=1)*k #- psi51*k
    rhs_forcing = np.vstack((stress_bc_rhs,-kin_ode_zs,-kin_ode_zb))
    return rhs_forcing



    
