""" This script was written to analyze chunk-averaged data.
"""

import numpy as np

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
            
    def find_crack_tip(self, step, threshold, window, ignore_last):
        """Cracktip at time step 
        """
        from . import pooling1d
        
        pad_size = (window - 1) // 2
        
        positions = self.find_local("Coord1", step=step)[:-ignore_last]
        edgeatoms = self.find_local("v_edgeatom", step=step)[:-ignore_last]
        edgeatoms = pooling1d(edgeatoms, window=window, pad_size=pad_size, mode='min')
        
        # Find crack tip
        crackbin = (edgeatoms>threshold).nonzero()[0][-1]
        cracktip = positions[crackbin]
        return cracktip
            
    def find_crack_tips(self, threshold, window, ignore_last=1):
        """Determine the crack tip position and speed
        using the number of edge atoms and a threshold.
        """
        
        self.timesteps = self.find_global("Timestep")
        self.cracktip = np.zeros_like(self.timesteps)
        
        for i in range(len(self.timesteps)):
            self.cracktip[i] = self.find_crack_tip(i, threshold, window, ignore_last)
        
        return self.cracktip
        
    def estimate_crack_speed(self, ignore_first=0, ignore_last=0, length_threshold=0.95):
        """Estimate average crack speed. If the crack propagates through
        the entire sample, the end time is when the crack has reached the
        end of the sample. The speed is returned in units Å/timestep
        """
        
        end_index = len(self.timesteps) - 1 #ignore_last
        positions = self.find_local("Coord1", step=0)
        length_sample = positions[len(positions) - ignore_last - 1]
            
        #try:
        #    g = self.cracktip[0]
        #else:
        #    raise ValueError("find_crack_tips needs to be run before this function")
            
        start_time = self.timesteps[ignore_first]
        end_time = self.timesteps[end_index]
        
        start_pos = self.cracktip[ignore_first]
        end_pos = self.cracktip[end_index]
        
        end = length_sample * length_threshold

        if end_pos > end:
            end_index = np.searchsorted(self.cracktip, end) + 1
            end_time = self.timesteps[end_index]

        speed = (end_pos - start_pos) / (end_time - start_time)
        return speed
        

    def plot_edge_fraction(self, step=0, show=False, save=False, ignore_first=3, ignore_last=10, threshold=0.004, window=1):
        """Given a chunk average file with the number of atoms of coordination number 1, 
        this function plots the edge fraction at a given timestep.
        """
        edgeatoms = self.find_local("v_edgeatom", step=step)
        positions = self.find_local("Coord1", step=step)
        timesteps = self.find_global("Timestep")
        current_timestep = int(timesteps[step])
        
        cracktip = self.find_crack_tip(step, threshold, window, ignore_last)

        import matplotlib.pyplot as plt

        label_size = {"size":14}                # Dictionary with size
        #plt.style.use("bmh")                    # Beautiful plots
        plt.rcParams["font.family"] = "Serif"   # Font
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.figure()
        plt.plot(positions[ignore_first:-ignore_last], edgeatoms[ignore_first:-ignore_last])
        plt.title(f"Timestep: {current_timestep}", **label_size)
        plt.axvline(cracktip, linestyle="--", color="r")
        plt.xlabel("Position in x-direction", **label_size)
        plt.ylabel("Fraction of surface atoms", **label_size)
        if save is not False:
            plt.savefig(save + f"edge_fraction_{current_timestep}.png")
        if show is not False:
            plt.show()
