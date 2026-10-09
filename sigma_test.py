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
from mc_xv import generate_xv_at,generate_xv
np.seterr(divide='ignore', invalid='ignore')

import corner

kappa=3.54
beta=3.47

def N_encounter_new(t,xv, b_max=1e-4): #stellar_rho_total use the better model
    return np.trapezoid(stellar_rho_total(xv[:,0], xv[:,1], xv[:,2]) * np.linalg.norm(xv[:, 3:],axis=1) * np.pi  * b_max**2,t)

def stellar_energy(N,M,R,v,its_sum):
    b=np.random.uniform(100*R, 1e-4,N) #is this a good choice?
    E=((4*agama.G*agama.G*M*0.27*R*R)/((b**4)*v**2)*3)
    E_sum=np.sum(E)
    E_rms_sum=np.sum(np.sqrt(E))**2
    if its_sum:
        return E_sum
    else:
        return E_rms_sum

for i in range(5):  
    print(i)  

    mc = Minicluster(mass=get_mass(), delta=50, concentration=10)
    Nbody_total=1e5
    pmass=mc.mass/Nbody_total


    simulation_time=10
    flag='alive'
    #final_pos=np.array([8.0,1.0,0.0,5.0,5.0,220.0])
    #mc_center=reverse_orbit(ic=final_pos,pot= pot_mw,time=simulation_time,timestart=0,show_start=0)
    s_pos=[8.0,0.0,0.0]  #end at solar_pos, but v is different , hence e is different
    final_pos=generate_xv_at(s_pos,1)[0]
    mc_center=reverse_orbit(ic=final_pos,pot= pot_mw,time=simulation_time,timestart=0,show_start=0)
    print(mc_center)
    
    
    mc_start=mc_center.copy()
    mc_end=final_pos.copy()
    mcp_xv,mcp_mass=np.empty((0, 6)),np.array([])     
    

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

    color_index=0
    
    SD_count=0
    
    while time_i < simulation_time:

        mc_time_center, mc_orbit_center = agama.orbit(ic=mc_center, potential=pot_mw, time=tupd,timestart=time_i,dtype=float, trajsize=1000, accuracy=1e-16,verbose=False)

        N_encounter_with_star=N_encounter_new(mc_time_center, mc_orbit_center,b_max=0.0001)
        N_encounter_with_star_array.append(N_encounter_with_star)
        col_time=tupd/N_encounter_with_star
        its_sum=col_time<dynamical_time(mc_mass,mc_rvir)
        delE=stellar_energy(round(N_encounter_with_star),mc_mass,mc_rvir, np.mean(np.linalg.norm(mc_orbit_center[:,3:],axis=1)),its_sum)
        

        Ei=binding_energy(mc_mass,mc_rvir)
        delEbyE=abs(delE/Ei)
        mass_loss=mc_mass*get_dm_frac(delEbyE)

        Ef=total_energy(Ei)+(1-get_fej(delEbyE))*delE+get_fub(delEbyE)*total_energy(Ei)
        Ef_array.append(Ef)
        Ei_array.append(total_energy(Ei))
        E2_array.append((1-get_fej(delEbyE))*delE)
        E3_array.append(get_fub(delEbyE)*total_energy(Ei))
        
        mc_mass=mc_mass-mass_loss
        mc_rvir=(kappa/2-beta)*agama.G*mc_mass*mc_mass/Ef if N_encounter_with_star > 1 else mc_rvir     #mass and radius update
        

        mass_array.append(mc_mass)
        radius_array.append(mc_rvir)
        mc_array.append(mc_orbit_center)
        mc_time.append(mc_time_center[-1])

        r_tidal=tidal_radius(np.linalg.norm(mc_center[:3]),mc_mass)
        pnum=round(mass_loss*Nbody_total/mc.mass)
        pnum_array.append(pnum)

        if mc_rvir==0:
            mcp_xv_temp,mcp_mass_temp=generate_mc_particles(mc_mass,r_tidal,pnum,pmass)
            mcp_xv_temp+=np.mean(mcp_xv[:,:6],axis=0)
            mcp_mass=np.concatenate([mcp_mass, mcp_mass_temp], axis=0)
            mcp_xv=np.concatenate([mcp_xv, mcp_xv_temp], axis=0)
            time_left=simulation_time-time_i
            mcp_xv = np.vstack(agama.orbit(ic=mcp_xv, potential=pot_mw,time=time_left, timestart=time_i, trajsize=1, accuracy=1e-16,dtype=float,verbose=False)[:, 1])
            mcp_xc=mcp_xv[:,:3]-mcp_xv[:,:3].mean(axis=0)
            pca=PCA(n_components=3)
            mcp_xr=pca.fit_transform(mcp_xc)
            SD.append(pca.explained_variance_**0.5)
            SD_t.append(simulation_time)
            SD_r.append(pca.explained_variance_ratio_)
            flag='dead'
            break
        


        
        

        
        
        if mcp_xv.size>0:
            mcp_xv = np.vstack(agama.orbit(ic=mcp_xv, potential=pot_mw,time=tupd, timestart=time_i, trajsize=1, accuracy=1e-16,dtype=float,verbose=False)[:, 1])
        
        mcp_xv_temp,mcp_mass_temp=generate_mc_particles(mc_mass,r_tidal,pnum,pmass)
            #if mcp_xv.size>0: mcp_xv_temp+=np.mean(mcp_xv[:,:6],axis=0) 
        if mcp_xv.size>0:mcp_xv_temp+=mc_orbit_center[-1]
        else: mcp_xv_temp+=mc_orbit_center[-1]  
        mcp_mass=np.concatenate([mcp_mass, mcp_mass_temp], axis=0)
        mcp_xv=np.concatenate([mcp_xv, mcp_xv_temp], axis=0)
    
        if pnum>0:
            color=np.concatenate([color,np.ones_like(mcp_mass_temp)*color_index])
            color_index+=1 

        mc_center = mc_orbit_center[-1]

        time_i += tupd
        #SVT
        if mcp_xv.size>1000 and SD_count%1==0:
            mcp_xc=mcp_xv[:,:3]-mcp_xv[:,:3].mean(axis=0)
            pca=PCA(n_components=3)
            mcp_xr=pca.fit_transform(mcp_xc)

            SD.append(pca.explained_variance_**0.5)
            SD_t.append(time_i)
            SD_r.append(pca.explained_variance_ratio_)
        SD_count+=1

    color=np.array(color[1:])


    orb=agama.orbit(ic=mc_start, potential=pot_mw, time=simulation_time, timestart=0, dtype=object, accuracy=1e-10, verbose=False)
    t_array=np.linspace(0,simulation_time,1000)
    orbit_array=orb(t_array)

    np.savez(f"sigma_data/mc{i}_{flag}.npy",
            mcp_xv=mcp_xv,
            mcp_mass=mcp_mass,
            mc_mass=mc.mass,
            mc_radius=mc.radius,
            color=color,
            mc_start=mc_start,
            mc_end=mc_center,
            t_array=t_array,
            orbit_array=orbit_array,
            SD=SD,
            SD_t=SD_t,
            SD_r=SD_r,
            mass_array=mass_array,
            radius_array=radius_array
            )


