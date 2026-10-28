

import numpy as np
import agama
agama.setUnits(length=1, velocity=1, mass=1)  # kpc, km/s, Solar mass
from scipy.stats import uniform_direction

def generate_mc_particles(M_f,R_f, N,pmass,M_i,R_i,delE,r_t):    
    m_array=np.ones(N)*pmass
    pos_unit=uniform_direction(3).rvs(N)  # shape (N, 3)
    coords=r_t*pos_unit
    vel_unit = uniform_direction(3).rvs(N)  # shape (N, 3)
    
    E=(total_energy(binding_energy(M_f,R_f))-total_energy(binding_energy(M_i,R_i))+delE)/pnum
    v_ideal=np.sqrt(2*E)
    vel=v_ideal*vel_unit
    return np.column_stack([coords, vel]), m_array

