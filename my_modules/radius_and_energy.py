import agama
agama.setUnits(length=1, velocity=1, mass=1)  # kpc, km/s, Solar mass
kappa=3.54
beta=3.47

pot_mw = agama.Potential('/suphys/aani0116/minicluster/minicluster_tidal/Agama/data/McMillan17.ini')

def tidal_radius(R,m):
    return R*(m/(3*pot_mw.enclosedMass(R)))**(1/3)

binding_energy = lambda mass, radius: beta * agama.G * mass**2 / radius
total_energy = lambda Eb: (kappa/2-beta) * Eb/beta
