
import numpy as np
import agama
agama.setUnits(length=1, velocity=1, mass=1)  # kpc, km/s, Solar mass
from scipy.stats import uniform_direction

def generate_mc_particles(M,R, N,pmass):
    
    m_array=np.ones(N)*pmass
    pos_unit=uniform_direction(3).rvs(N)  # shape (N, 3)
    coords=R*pos_unit
    vel_unit = uniform_direction(3).rvs(N)  # shape (N, 3)
    v_ideal=np.sqrt(agama.G*M/R)
    vel=v_ideal*vel_unit*0
    return np.column_stack([coords, vel]), m_array

