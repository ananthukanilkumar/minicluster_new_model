import numpy as np
import agama
agama.setUnits(length=1, velocity=1, mass=1)  # kpc, km/s, Solar mass
import matplotlib.pyplot as plt
from IPython.display import Latex
import sys
sys.path.append('/suphys/aani0116/Work/minicluster_model_new/my_modules/')
from scipy.stats import uniform_direction
from reverse_orbit import reverse_orbit
from mc_declare import *
pot_mw = agama.Potential('/suphys/aani0116/minicluster/minicluster_tidal/Agama/data/McMillan17.ini')
sys.path.append('/suphys/aani0116/Work/minicluster_tidal/stellar')
from N_calculator import N_encounter 
from N_calculator import stellar_rho_total
from scipy.integrate import quad
from hmf import get_mass
from times import dynamical_time
from mass_loss_new import *
from generate_particles import *
from radius_and_energy import *
from sklearn.decomposition import PCA
import corner
from mc_xv import generate_xv_at,generate_xv

kappa=3.54
beta=3.47
s_pos=[8.0,0.01,0.01]
R= np.linalg.norm(np.array(s_pos))
def stellar_energy(N,M,R,v,its_sum):
    b = np.sqrt(np.random.uniform(0, (1e-3)**2, N))
    E=((4*0.33*agama.G*agama.G*M*0.27*R*R)/((b**4)*v**2))
    E_sum=np.sum(E)
    E_rms_sum=np.sum(np.sqrt(E))**2
    if its_sum:
        return E_sum
    else:
        return E_rms_sum

def N_encounter_new(t,xv, b_max=1e-3): #stellar_rho_total use the better model
    return np.trapezoid(stellar_rho_total(xv[:,0], xv[:,1], xv[:,2]) * np.linalg.norm(xv[:, 3:],axis=1) * np.pi  * b_max**2,t)

vcirc=lambda R:np.sqrt(-R * pot_mw.force(np.column_stack((R, 0*R, 0*R)))[:, 0])
f = lambda E: np.exp(E / sigma**2)   

sigma = vcirc(R)/np.sqrt(2)                     
vesc  = np.sqrt(-2*pot_mw.potential(s_pos))                     
Psi   = pot_mw.potential(s_pos)
N     = 1000


v = np.linspace(0, vesc, 100000)
P = v**2 * f(Psi - v**2/2)
P /= P.sum()

get_speed = lambda N: np.random.choice(v, size=N, p=P)

def get_xv(pos):
    v=get_speed(1)[0]
    vel_unit = uniform_direction(3).rvs(1)
    vel=v * vel_unit
    return np.hstack((pos, vel[0]))

SD_1,SD_2,SD_3,minicluster_mass,minicluster_radius,minicluster_start_pos=[],[],[],[],[],[]
print('the job has',200,'mc and',1e5,'particles')
for i in range(200):
    dead_flag=0
    print(i)
    simulation_time=10
    mcp_xv,mcp_mass=np.empty((0, 6)),np.array([]) 
    mc = Minicluster(mass=get_mass(), delta=10, concentration=10)
    Nbody_total=1e5
    pmass=mc.mass/Nbody_total
    #print('mass and radius:',mc.mass,mc.radius,)
    mc_center=get_xv(s_pos)
    mc_start  = mc_center.copy()
    tupd = 1e-2
    time_i = 0
    mc_mass=mc.mass
    mc_rvir=mc.radius
    #random stuff for plotting
    mc_array=[mc_center]
    mc_time=[]
    pnum_array=[]
    mass_array,radius_array=[],[]
    Ef_array,Ei_array,E2_array,E3_array,N_encounter_with_star_array =[],[],[],[],[]
    color=[0]
    SD=[]
    SD_r=[]
    SD_t=[]
    SD_axis=[]
    R_vector=[]
    color_index=0
    SD_count=0
    #print(f'enter loop {i}')
    minicluster_mass.append(mc.mass)
    minicluster_radius.append(mc.radius)

    while time_i < simulation_time:
        mc_time_center, mc_orbit_center = agama.orbit(ic=mc_center, potential=pot_mw, time=tupd,timestart=time_i,dtype=float, trajsize=1000, accuracy=1e-16,verbose=False)
        N_encounter_with_star=N_encounter_new(mc_time_center, mc_orbit_center,b_max=1e-3)
        N_encounter_with_star_array.append(N_encounter_with_star)
        col_time=tupd/N_encounter_with_star
        its_sum=col_time<dynamical_time(mc_mass,mc_rvir)
        
        delE=stellar_energy(round(N_encounter_with_star),mc_mass,mc_rvir, np.mean(np.linalg.norm(mc_orbit_center[:,3:],axis=1)),its_sum)
        Eb=binding_energy(mc_mass,mc_rvir)
        Ei=total_energy(Eb)
        delEbyE=abs(delE/Eb)
        mass_loss=mc_mass*get_dm_frac(delEbyE)
        Ef=Ei+(1-get_fej(delEbyE))*delE-get_fub(delEbyE)*Ei
        
        
        mc_mass_old=mc_mass
        mc_mass=mc_mass-mass_loss
        mc_rvir=((kappa/2-beta)*agama.G*mc_mass*mc_mass/Ef) if Ef<0 else 0
        r_tidal=tidal_radius(np.linalg.norm(mc_center[:3]),mc_mass_old)
        pnum=round(mass_loss*Nbody_total/mc.mass)
        pnum_array.append(pnum)

        if mc_rvir==0:
            print('dead')
            pnum=round(mc_mass_old*Nbody_total/mc.mass)        
            mcp_xv_temp,mcp_mass_temp=generate_mc_particles(mc_mass_old,r_tidal,pnum,pmass)
            mcp_xv_temp+=mc_center
            mcp_xv=np.concatenate([mcp_xv, mcp_xv_temp], axis=0)
            mcp_xv = np.vstack(agama.orbit(ic=mcp_xv, potential=pot_mw,time=simulation_time, timestart=time_i, trajsize=1, accuracy=1e-16,dtype=float,verbose=1)[:, 1])
            mcp_xc=mcp_xv[:,:3]-mcp_xv[:,:3].mean(axis=0)
            pca=PCA(n_components=3)
            mcp_xr=pca.fit_transform(mcp_xc)
            SD.append(pca.explained_variance_**0.5)
            SD_t.append(time_i)
            alive=0
            break
    
        if mcp_xv.size>0:
            mcp_xv = np.vstack(agama.orbit(ic=mcp_xv, potential=pot_mw,time=tupd, timestart=time_i, trajsize=1, accuracy=1e-16,dtype=float,verbose=False)[:, 1])
        
        mcp_xv_temp,mcp_mass_temp=generate_mc_particles(mc_mass,r_tidal,pnum,pmass)
        mcp_xv_temp+=mc_orbit_center[-1]  
        mcp_xv=np.concatenate([mcp_xv, mcp_xv_temp], axis=0)

        mc_center = mc_orbit_center[-1]

        time_i += tupd
        if mcp_xv.size>1000 and SD_count%1==0:
            mcp_xc=mcp_xv[:,:3]-mcp_xv[:,:3].mean(axis=0)
            pca=PCA(n_components=3)
            mcp_xr=pca.fit_transform(mcp_xc)
            SD.append(pca.explained_variance_**0.5)
            SD_t.append(time_i)
        SD_count+=1
        print(time_i)

   
    if mcp_xv.size>100 and dead_flag==0:
        SD=np.array(SD)
        SD_1.append(SD[-5:,0].mean())
        SD_2.append(SD[-5:,1].mean())
        SD_3.append(SD[-5:,2].mean())
    elif mcp_xv.size>100 and dead_flag==1:
        SD=np.array(SD)
        SD_1.append(SD[-1,0].mean())
        SD_2.append(SD[-1:,1].mean())
        SD_3.append(SD[-1:,2].mean())
    else:
        SD_1.append(0)
        SD_2.append(0)
        SD_3.append(0)
    
        
    
    

    #print(i,'done','SD',SD_1[-1],SD_2[-1],SD_3[-1])
    
i+=1

np.savez('minicluster_results200_p1e5.npz',
         SD_1=np.array(SD_1), SD_2=np.array(SD_2), SD_3=np.array(SD_3),
         minicluster_mass=np.array(minicluster_mass),
         minicluster_radius=np.array(minicluster_radius),
         minicluster_start_pos=np.array(minicluster_start_pos))
