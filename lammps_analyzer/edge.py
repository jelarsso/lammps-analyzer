""" The Edge class can be used to determining the crack tip position of 
a fracture simulation. 
"""

import numpy as np
import matplotlib.pyplot as plt

label_size = {"size":14}                # Dictionary with size
plt.style.use("bmh")                    # Beautiful plots
plt.rcParams["font.family"] = "Serif"   # Font
plt.rcParams.update({'figure.max_open_warning': 0})

class Edge:
    def __init__(self, filename, threshold=0.001, window=1):
        # Declare lists
        self.chunk_list_tot = []
        self.pos_list_tot = []
        self.num_atoms_tot = []
        self.edge_fraction_tot = []
        self.crack_tips = []
        self.timesteps = []
        
        self.find_crack_tip(filename, threshold, window)

    def find_crack_tip(self, filename, threshold, window):
        """Determine the crack tip position and speed
        using the number of edge atoms and a threshold.
        """
        
        pad_size = (window - 1) // 2
        
        with open(filename, "r") as f:
            
            # Declare integers
            current_timestep=0
            num_chunks=0
            crack_tip = 0
            
            # Declare local lists
            chunk_list = []
            pos_list = []
            num_atoms = []
            edge_fraction = []
                
            for line in f.readlines():
                splitted = line.split()
                if line.startswith("#"):
                    continue
                elif len(splitted) == 3:
                    # Append to global lists
                    self.chunk_list_tot.append(chunk_list)
                    self.pos_list_tot.append(pos_list)
                    self.num_atoms_tot.append(num_atoms)
                    self.edge_fraction_tot.append(edge_fraction)
                    
                    # Update current timestep
                    current_timestep = int(splitted[0])
                    num_chunks = int(splitted[1])
                    
                    # Transform lists to numpy arrays
                    pos_list = np.array(pos_list, dtype=float)
                    edge_fraction = np.array(edge_fraction, dtype=float)
                    edge_fraction = self.pooling(edge_fraction, window=window, pad_size=pad_size, mode='min')
                    
                    # Find crack tip
                    try:
                        crack_bin = (edge_fraction[:-10]>threshold).nonzero()[0][-1]
                        crack_tip = pos_list[crack_bin]
                    except:
                        pass
                    self.crack_tips.append(crack_tip)
                    self.timesteps.append(current_timestep)
                    
                    # Redeclare local lists
                    chunk_list = []
                    pos_list = []
                    num_atoms = []
                    edge_fraction = []
                    
                elif len(splitted) == 4:
                    # Append values to lists
                    chunk_list.append(splitted[0])
                    pos_list.append(splitted[1])
                    num_atoms.append(splitted[2])
                    edge_fraction.append(splitted[3])
                    

        self.crack_tips = np.array(self.crack_tips, dtype=float)
        self.timesteps = np.array(self.timesteps, dtype=int)
        self.edge_fraction = self.edge_fraction_tot
        self.timesteps = self.timesteps
        self.pos_list = self.pos_list_tot
        
    @staticmethod
    def average(arr, window):
        remainder = len(arr) % window
        avg = np.mean(arr[:-remainder].reshape(-1, window), axis=1)
        return avg
        
    @staticmethod
    def pooling(arr, window, pad_size=0, stride=1, mode='min'):
        # Padding
        if mode == 'min':
            A = np.full(len(arr) + 2 * pad_size, np.inf)
        elif mode == 'max':
            A = np.full(len(arr) + 2 * pad_size, -np.inf)
        A[pad_size:len(arr) + pad_size] = arr
        
        # Window view of data
        from numpy.lib.stride_tricks import as_strided
        output_shape = ((len(A) - window)//stride + 1,)
        A_w = as_strided(A, shape = output_shape + (window,), 
                            strides = (stride*A.strides) + A.strides)
        
        if mode == 'max':
            return A_w.max(axis=1)
        elif mode == 'min':
            return A_w.min(axis=1)
        elif mode == 'avg' or mode == 'mean':
            return A_w.mean(axis=1)
        else:
            raise NotImplementedError("Mode {} is not implemented".format(mode))
        

    def plot_edge_fraction(self, ignore_first=5, ignore_last=10, plot_every=np.inf):
        from tqdm import trange
        for i in trange(len(self.edge_fraction)):
            edge_fraction = np.array(self.edge_fraction[i], dtype=float)
            pos_list = np.array(self.pos_list[i], dtype=float)
            current_timestep = int(self.timesteps[i])
            crack_tip = self.crack_tips[i]
            if i % plot_every == 0:
                # Plot 
                plt.figure()
                plt.plot(pos_list[ignore_first:-ignore_last], edge_fraction[ignore_first:-ignore_last])
                plt.title(f"Timestep: {current_timestep}", **label_size)
                plt.axvline(crack_tip, linestyle="--", color="r")
                plt.xlabel("Block position in x-direction", **label_size)
                plt.ylabel("Fraction of edge atoms", **label_size)
                #plt.savefig(f"../data/stretch3_wedgeBeta_notch_defects_576000/fig/edge_fraction_pooling/edge_fraction_{current_timestep}.png")
                #plt.show()
        
    
        
    def plot_tip_position(self, ignore_first=5):
        plt.figure()
        plt.plot(self.timesteps[ignore_first:], self.crack_tips[ignore_first:])
        plt.title("Crack tip motion in x-direction")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.savefig("../fig/crack_tip_position.png")
        plt.plot()

    def plot_tip_speed(self, ignore_first=5):
        plt.figure()
        plt.plot(self.timesteps[ignore_first+1:], np.diff(self.crack_tips[ignore_first:]))
        plt.xlabel("Time")
        plt.ylabel("Speed")
        plt.savefig("../fig/crack_tip_speed.png")
        plt.plot()
        
    
if __name__ == "__main__":
    density = Edge("../data/stretch3_wedgeBeta_notch_defects_576000/edge_density.data", threshold=0.001, window=7)
    density.plot_edge_fraction()
    crack_tip = density.crack_tips
    timesteps = density.timesteps
    
    dt = 0.002
    
    crack_tip50 = density.average(crack_tip, 50)
    speed200 = density.average(np.diff(crack_tip, prepend=10), 200) * 100
    timesteps50 = density.average(timesteps, 50)
    timesteps200 = density.average(timesteps, 200)
    
    time50 = timesteps50 * dt
    time200 = timesteps200 * dt
    
    plt.figure()
    #plt.subplot(2,1,1)
    plt.plot(time50, crack_tip50)
    plt.xlabel("Time [ps]")
    plt.ylabel("Position [Å]")
    
    plt.figure()
    #plt.subplot(2,1,2)
    plt.plot(time200, speed200)
    plt.xlabel("Time [ps]")
    plt.ylabel("Speed [m/s]")
    plt.show()
