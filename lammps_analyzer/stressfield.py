from lammps_analyzer.chunkavg import ChunkAvg
import numpy as np
import matplotlib.pyplot as plt

class Stressfield():

    """
    Class for reading chunkavg files of the stressfield sigma_ij from LAMMPS.
    """


    def __init__(self, filename):
        self.data = ChunkAvg(filename)
        self.component_ordering = {"xx":1, "yy":2, "zz":3, "xy":4} #mapping between the index of the tensor array and component

    
    def _get_axes(self, step=0):
        coordx = self.data.find_local("Coord1",step=step)
        coordy = self.data.find_local("Coord2",step=step)

        #Count how many bins in each dim
        i = np.size(np.unique(coordx))
        j = np.size(np.unique(coordy))
        
        assert i*j == self.data.find_global("Number-of-chunks")[step], "Error in counting bins per axis"
        
        mesh_XX = np.resize(coordx,(i,j))
        mesh_YY = np.resize(coordy,(i,j))

        nbins_X = i
        nbins_Y = j
        return mesh_XX,mesh_YY,nbins_X,nbins_Y


    def create_stressfield(self, step=0, component="xx"):
        
        mesh_XX,mesh_YY,nbins_X,nbins_Y = self._get_axes(step=step)
        component_index = self.component_ordering[component]

        stressfield = self.data.find_local(f"c_stressperatom[{str(component_index)}]",step=step)
        stressfield = np.resize(stressfield,(nbins_X,nbins_Y))
        
        self.stress = (mesh_XX,mesh_YY,stressfield,nbins_X,nbins_Y,component,step)
        return stressfield
    
    def create_stressfields(self,component="xx"):
        mesh_XX,mesh_YY,nbins_X,nbins_Y = self._get_axes(step=0)

        self.timesteps = self.data.find_global("Timestep")
        self.stressfields = np.zeros((self.timesteps.size,nbins_X,nbins_Y))

        for t in range(self.timesteps):
            self.stressfields[t] = self.create_stressfield(step=t,component=component)
        
        return mesh_XX,mesh_YY,self.timesteps, self.stressfields


    def contourplot_stressfield(self,logplot=False,stress=None,show=False,save=False):
        """
        If you want to save, save should be the string of the path/filename of desired output.
        If stress is None, this plots the stressfield made by the previous create_stressfield call.
        """
        if stress is None:
            assert hasattr(self,"stress"), "create_stressfield must be called first or stress field passed as arg!"
            stress = self.stress
        
        mesh_XX,mesh_YY,stressfield,nbins_X,nbins_Y,component,step = stress

        if logplot:
            stressfield=np.where(stressfield!=0,np.log(np.abs(stressfield)),0)
            cbartitle = "stress field value [log]"
        else:
            cbartitle = "stress field value"

        plt.contourf(mesh_XX,mesh_YY,stressfield)
        plt.title(f"Stress field $\\sigma_{{{component}}}$ nbins = {nbins_X*nbins_Y} at timestep = {step}")
        plt.xlabel("x-length [Å]")
        plt.ylabel("y-length [Å]")
        cbar  = plt.colorbar()
        cbar.set_label(cbartitle)

        if save:
            plt.savefig(save)
        if show:
            plt.show()
    
    def imshow_stressfield(self,logplot=False,stress=None,show=False,save=False,cmap="bwr"):
        """
        If you want to save, save should be the string of the path/filename of desired output.
        If stress is None, this plots the stressfield made by the previous create_stressfield call.
        """
        if stress is None:
            assert hasattr(self,"stress"), "create_stressfield must be called first or stress field passed as arg!"
            stress = self.stress
        
        mesh_XX,mesh_YY,stressfield,nbins_X,nbins_Y,component,step = stress

        if logplot:
            stressfield=np.where(stressfield!=0,np.log(np.abs(stressfield)),0)
            cbartitle = "stress field value [log]"
        else:
            cbartitle = "stress field value"

        vmin,vmax = -np.max(np.abs(stressfield)),np.max(np.abs(stressfield))

        plt.imshow(stressfield.T,cmap=cmap,origin="lower",vmin=vmin, vmax=vmax, extent=[np.min(mesh_XX),np.max(mesh_XX),np.min(mesh_YY),np.max(mesh_YY)])
        plt.title(f"Stress field $\\sigma_{{{component}}}$ nbins = {nbins_X*nbins_Y} at timestep = {step}")
        plt.xlabel("x-length [Å]")
        plt.ylabel("y-length [Å]")
        cbar = plt.colorbar()
        cbar.set_label("value of stress field [pressure*volume]")
        if save:
            plt.savefig(save)
        if show:
            plt.show()
        
    
    def avg_tip(self,edge_density,window_stress=10,component="xy"):
        """
        Create the average around the cracktip
        """
        
        cracktip_positions = edge_density.find_crack_tips(threshold=0.004,window=1,ignore_last=100)

        mesh_XX, mesh_YY, timesteps, stressfields = self.create_stressfields(component=component)

        xaxis = mesh_XX[:,0]

        stress_avg = np.zeros((2*window_stress,mesh_YY[0,:].size))

        for t,cracktip in enumerate(cracktip_positions):
            xindice = np.argmin(np.abs(xaxis-cracktip))
            stress_avg += stressfields[t,(xindice-window_stress):(xindice+window_stress),:]
        
        return stress_avg/cracktip_positions.size






