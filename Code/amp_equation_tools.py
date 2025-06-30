"""
Macro functions to solve for amplitude equation
"""

import numpy as np
from importlib import reload


#import matplotlib
#matplotlib.use("TkAgg")
#import matplotlib.pyplot as plt
#plt.ion()
#plt.isinteractive()

import sys
sys.path.append("../Code/")

import stress_stream
reload(stress_stream)
import stress_stream as stream
import non_hydro
reload(non_hydro)
import non_hydro as model



def solveAmpEqn(k,ms=0,mb=0.0,n=1):
    # Determine critical stability parameter
    S0 = stream.Scrit(k,ms=0,mb=mb)

    # Eigenvalues and eigenvector for leading order mode
    #eig,vec = stream.eigs(k,S0)

    # Create class for leading order term
    h11=stream.StreamFun(k,order=1,ms=ms,mb=mb)

    # Solve leading order term for coefficients
    h11.solve(S0,rhs=None)

    # Create a running dictionary of terms
    funcs={'h11':h11}


    # Solve for second order term psi22
    h22=stream.StreamFun(k,order=2,ms=ms,mb=mb)
    rhs22 = model.rhs22(k,S0,funcs)
    h22.solve(S0,rhs22)

    funcs['h22']=h22

    # Second order mean field term, this term is hydrostatic
    h20=stream.StreamFun(k,order=0,ms=ms,mb=mb)
    rhs20=model.rhs20(k,S0,funcs)
    h20.solve(S0,rhs20)
    h20.stress=-4*S0*h20.h
    funcs['h20']=h20


    # Third order term psi33
    h33=stream.StreamFun(k,order=3,ms=ms,mb=mb)
    rhs33=model.rhs33(k,S0,funcs)
    h33.solve(S0,rhs33)
    funcs['h33']=h33


    # Third order term psi31, needs to be treated differently
    rhs31=model.rhs31(k,S0,funcs)
    h31=stream.StreamFun(k,order=1,ms=ms,mb=mb)
    S2=h31.solveS(S0,rhs31,funcs,S2=0.0)
    h31.set_amp_eqn(S0,rhs=rhs31,funcs=funcs,S2=None)
    h31.s,h31.b=0,0
    funcs['h31']=h31

    rhs42=model.rhs42(k,[S0,S2],funcs)
    h42=stream.StreamFun(k,order=2,ms=ms,mb=mb)
    h42.solve(S0,rhs42)
    funcs['h42']=h42

    rhs40=model.rhs40(k,[S0,S2],funcs)
    h40=stream.StreamFun(k,order=0,ms=ms,mb=mb)
    h40.solve(S0,rhs40)
    h40.stress = -4*S0*h40.h - 4*S2*h20.h
    funcs['h40']=h40


    rhs51=model.rhs51(k,[S0,S2],funcs)
    #part_sol = stream.ParticularSol(h11)
    #stress_bc = part_sol.stress_rhs(n=3)
    #kin_bc = part_sol.kinematic_rhs(n=3)
    #rhs51_non_newt = part_sol.non_newtonian_rhs(n=1e6)
    #print(rhs51)
    #rhs51 = rhs51 + rhs51_non_newt
    #print(rhs51)

    h51=stream.StreamFun(k,order=1,ms=ms,mb=mb)
    S4=h51.solveS(S0,rhs51,funcs,S2=S2)
    #print(h11.C)
    #print("S4",S4)
    #funcs['h51']=h51


    # Fourth order term psi42


    # Mean field term psi40, this term is hydrostatic 
    amp_eqn= stream.AmplitudeEquation(S=[S0,S2,S4],h11=h11,rhs31=rhs31,rhs51=rhs51,n=n)
    return [S0,S2,S4],funcs,amp_eqn

def evolve(k,S,A,t,dt,melt_rate=0.0):
        [S0,S2,S4],funcs,amp_eqn=solveAmpEqn(k)
        dAdt = amp_eqn(S,A)
        A = A + (dAdt+melt_rate*A)*dt 
        k = k - 2*dt*k
        t = t + dt
        return k,A,t

