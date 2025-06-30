"""
Calculate rhs of amplitude equation
"""
import numpy as np
from importlib import reload
import sys
sys.path.append("../Code/")
import amp_equation_tools;reload(amp_equation_tools);import amp_equation_tools as amp
import stress_stream;reload(stress_stream);import stress_stream as stream

def rhs(dt,k,A,S0,bottom_melt,thick,strain_rate):
    Scrit,funcs,amp_eqn=amp.solveAmpEqn(k,ms=0.0,mb=bottom_melt/(thick*strain_rate),n=3)
    dAdt = amp_eqn(S=S0,A=A,order=5)
    A += dAdt*dt*strain_rate
    damage =  np.array(stream.amp2dam(k,[A],funcs)).item()
    return A,damage 

    
    




