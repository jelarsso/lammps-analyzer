import numpy as np
import matplotlib.pyplot as plt
label_size = {"size":14}                # Dictionary with size
plt.style.use("bmh")                    # Beautiful plots
plt.rcParams["font.family"] = "Serif"   # Font

class Dump:
    """ Analyzing dump files, containing particle-specific information. 
    
    Parameters
    ----------
    filename : str
        which dump file to read
    info_lines : int
        number of lines storing information
    particle_line : int
        the line that gives the number of particles
    """
    def __init__(self, filename, info_lines=9, particle_line=3):
        print("Loading data. For large files, this might takes a while.")
        f = open(filename,'rt')
        self.particles = int(f.readlines()[particle_line])
        f.close()
        num_lines = sum(1 for line in open(filename))
        length = self.particles + info_lines
        self.steps = int(num_lines / length)
        data = []
        from tqdm import tqdm
        for t in tqdm(range(self.steps)):
            data.append(np.loadtxt(open(filename,'rt').readlines()[t * length + info_lines: (t+1) * length]))
        self.data = np.asarray(data)


    def plot_position_distribution(self, steps=[0], show=False, save=False):
        """ Plots a histogram of the radius of the particles. 
        
        Parameters
        ----------
        steps : list
            the timesteps of interest
        show : bool
            display plot yes/no (True/False). False by default
        save : bool
            save plot yes/no (True/False). False by default
        """
        for i in steps:
            positions = self.data[i][:, 2:5]
            radius = np.linalg.norm(positions, axis=1)
            plt.hist(radius, 100, density=True, facecolor='b', alpha=0.75)
            plt.xlabel('Radius')
            plt.ylabel('Density')
            plt.grid()
            if show: plt.show()
            if save: plt.savefig('../fig/position_distribution_{}.png'.format(t))


    def plot_velocity_distribution(self, steps=[0], show=False, save=False):
        """ Plots a histogram of the speed of all particles at selected timesteps.
        
        Parameters
        ----------
        steps : list
            the timesteps of interest
        show : bool
            display plot yes/no (True/False). False by default
        save : bool
            save plot yes/no (True/False). False by default
        """
        velocity = self.data[10:, :, 5:8]
        velocity = velocity.reshape(-1, velocity.shape[-1])
        speed = np.linalg.norm(velocity, axis=1)
        plt.hist(speed, 100, density=True, facecolor='b', alpha=0.75)
        plt.title("4000 argon atoms, NVE, dt=0.005")
        plt.xlabel(r'Speed [v/$\sigma/\tau$]', **label_size)
        plt.ylabel('Density', **label_size)
        plt.grid()
        if show: plt.show()
        if save: plt.savefig('../fig/velocity_distribution_{}.png'.format(t))
            
    def plot_diffusion(self, show=False, save=False):
        """ Plot the diffusion and estimate the diffusion constant.
        
        Parameters
        ----------
        show : bool
            display plot yes/no (True/False). False by default
        save : bool
            save plot yes/no (True/False). False by default
        """
        pos = self.data[:,:,2:5]
        initial_pos = pos[0]
        diff = pos - initial_pos[np.newaxis]
        res = np.einsum('ijk,ijk->i',diff,diff)
        plt.plot(res)
        if save: plt.savefig("../fig/diffusion.png")
        if show: plt.show()
        
        t = np.linspace(0, 500, len(res))
        D = np.divide(res, t) / 6
        print(D)
        
    def plot_radial(self, show=False, save=False, L=3):
        """ Plot the radial distribution function as a function of relative distance.
        
        Parameters
        ----------
        show : bool
            display plot yes/no (True/False). False by default
        save : bool
            save plot yes/no (True/False). False by default
        L : 
        """
        pos = self.data[:,:,2:5]
        from scipy.spatial import distance_matrix
        d = distance_matrix(pos, pos)
        d_ = d.flatten()
        bins = np.linspace(0, L/2, 1000)
        inds = np.digitize(d.flatten(), bins)
        plt.plot(inds)
        if show: plt.show()
       
        

