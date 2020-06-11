""" This script was written to analyze chunk-averaged data.
"""

import numpy as np
import matplotlib.pyplot as plt

label_size = {"size":14}                # Dictionary with size
plt.style.use("bmh")                    # Beautiful plots
plt.rcParams["font.family"] = "Serif"   # Font
plt.rcParams.update({'figure.max_open_warning': 0})

class ChunkAvg:
    
    def __init__(self, filename):
        self.read_data(filename)
        
    def read_data(self, filename):
        """Read chunk-averaged file. 
        
        Assumptions:
        The file starts with three comment lines. The first line is some 
        general information. The second line contains global information. 
        The third line contains local information. 
        """
        
        with open(filename, "r") as f:
        
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()
            
            self.global_labels = line2.split()[1:]
            self.local_labels = line3.split()[1:]
            
            global_list = []
            local_list = []
            
            for line in f.readlines():
                splitted = line.split()
                if line[0].isdigit():
                    # global
                    global_list.append(splitted)
                    local_list.append([])
                    
                elif line.startswith(" "):
                    # local
                    local_list[-1].append(splitted)
                    
                else:
                    raise TypeError("???")
            
        self.global_list = np.array(global_list, dtype=float).transpose(1,0)
        self.local_list = [np.array(i, dtype=float).transpose(1,0) for i in local_list]
        
    def global_labels(self):
        """Get an overview of all the global labels
        """
        return self.global_labels
        
    def local_labels(self):
        """Get an overview of all the local labels
        """
        return self.local_labels
        
    def find_global(self, key):
        """Returns the ...
        """
        array = None
        for i, variable in enumerate(self.global_labels):
            if variable == key:
                array = self.global_list[i]
        if array is None:
            raise KeyError("No category named {} found.".format(key))
        else:
            return np.array(array)
        
    def find_local(self, key, step=0):
        """Returns the ...
        """
        array = None
        for i, variable in enumerate(self.local_labels):
            if variable == key:
                array = self.local_list[step][i]
        if array is None:
            raise KeyError("No category named {} found.".format(key))
        else:
            return np.array(array)
            
    @staticmethod
    def average(arr, window):
        """Average an array arr over a certain window size
        """
        remainder = len(arr) % window
        avg = np.mean(arr[:-remainder].reshape(-1, window), axis=1)
        return avg
        
    @staticmethod
    def pooling1d(arr, window, pad_size=0, stride=1, mode='min'):
        """Perform 1d pooling on an array arr
        """
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
            
    def find_crack_tip(self, threshold, window):
        """Determine the crack tip position and speed
        using the number of edge atoms and a threshold.
        """
        
        pad_size = (window - 1) // 2
        
        self.timesteps = self.find_global("Timestep")
        cracktip = np.zeros_like(self.timesteps)
        
        self.positions_list = []
        self.edgeatoms_list = []
        
        for i, timestep in enumerate(self.timesteps):
            positions = self.find_local("Coord1", step=i)
            edgeatoms = self.find_local("v_edgeatom", step=i)
            edgeatoms = self.pooling1d(edgeatoms, window=window, pad_size=pad_size, mode='min')
            
            self.positions_list.append(positions)
            self.edgeatoms_list.append(edgeatoms)
            
            # Find crack tip
            try:
                crackbin = (edgeatoms[:-10]>threshold).nonzero()[0][-1] + pad_size
                cracktip[i] = positions[crackbin]
            except:
                pass
        
        return cracktip

    def plot_edge_fraction(self, plot_every=np.inf, show=False, save=False, ignore_first=5, ignore_last=10):
        """Given a chunk average file with the number of atoms of coordination number 1, 
        this function plots the edge fraction at a given timestep.
        """
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
            if save is not False:
                plt.savefig(save + f"edge_fraction_{current_timestep}.png")
            if show is not False:
                plt.show()
        
    
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
