import numpy as np

class LJ2Si:
    """ Convert Lennard-Jones units to Si-units
    """
    def __init__(self, m, ε, σ):
        self.m = m
        self.ε = ε
        self.σ = σ
        self.τ = np.sqrt(self.m * self.σ**2 / self.ε)
        self.kB = 1.38064852e-23 # m^2 kg s^-2 K^-1
        
    def length(self, length_LJ):
        length_LJ = np.asarray(length_LJ)
        return length_LJ * self.σ
        
    def area(self, area_LJ):
        area_LJ = np.asarray(area_LJ)
        return area_LJ * self.σ**2
        
    def volume(self, volume_LJ):
        volume_LJ = np.asarray(volume_LJ)
        return volume_LJ * self.σ**3
        
    def time(self, time_LJ):
        time_LJ = np.asarray(time_LJ)
        return time_LJ * self.τ
        
    def temp(self, temp_LJ):
        temp_LJ = np.asarray(temp_LJ)
        return temp_LJ * self.ε / self.kB
        
    def force(self, force_LJ):
        force_LJ = np.asarray(force_LJ)
        return force_LJ * self.ε / self.σ
        
    def energy(self, energy_LJ):
        energy_LJ = np.asarray(energy_LJ)
        return energy_LJ * self.ε
        
    def pressure(self, pressure_LJ):
        pressure_LJ = np.asarray(pressure_LJ)
        return pressure_LJ * self.ε / self.σ**3
        
    def surface_tension(self, tension_LJ):
        tension_LJ = np.asarray(tension_LJ)
        return tension_LJ * self.ε / self.σ**2
