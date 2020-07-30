from lammps_analyzer.chunkavg import ChunkAvg,ChunkAvgMemSave
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


class Stressfield():

    """
    Class for reading chunkavg files of the stressfield sigma_ij from LAMMPS.
    """


    def __init__(self, filename,dt=0.001, loc=".",loadfromnpy=False, xlength = 950, ylength = 121.5, ntimesteps = 50000):
        if loadfromnpy:
            self.loc = loc
            self.xlength = xlength
            self.ylength = ylength
            self.ntimesteps = ntimesteps
            self.load_from_npy()
            print("Loaded from .npy: beware some methods are not implemented for this initialization.")
        else:
            self.data = ChunkAvgMemSave(filename)
        self.component_ordering = {"xx":1, "yy":2, "zz":3, "xy":4} #mapping between the index of the tensor array and component
        self.loadfromnpy = loadfromnpy 
        self.dt = dt
    
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
        
        if not self.loadfromnpy:    
            mesh_XX,mesh_YY,nbins_X,nbins_Y = self._get_axes(step=step)
            component_index = self.component_ordering[component]

            stressfield = self.data.find_local(f"c_stressperatom[{str(component_index)}]",step=step)
            stressfield = np.resize(stressfield,(nbins_X,nbins_Y))
            
            self.stress = (mesh_XX,mesh_YY,stressfield,nbins_X,nbins_Y,component,step)
        elif self.loadfromnpy:
            mesh_XX,mesh_YY,timesteps,stressfields = self.load_from_npy(component=component)
            stressfield = stressfields[step]
            self.stress = (mesh_XX,mesh_YY,stressfield,mesh_XX[:,0].size,mesh_YY[0,:].size,component,step)
        return stressfield
    
    def create_stressfields(self,component="xx"):
        if not self.loadfromnpy:
            self.mesh_XX,self.mesh_YY,nbins_X,nbins_Y = self._get_axes(step=0)

            self.timesteps = self.data.find_global("Timestep")
            self.stressfields = np.zeros((self.timesteps.size,nbins_X,nbins_Y))

            for t in range(self.timesteps.size):
                self.stressfields[t] = self.create_stressfield(step=t,component=component)
            
            self.last_component_created = component
            return self.mesh_XX,self.mesh_YY,self.timesteps, self.stressfields
        else:
            return self.load_from_npy(component=component)




    def save_stressfields(self,loc="."):
        print("Saving.",end="")
        self.create_stressfields(component="xx")
        np.save(loc+"/stress_xx.npy",self.stressfields)
        print(".",end="")
        self.create_stressfields(component="xy")
        np.save(loc+"/stress_xy.npy",self.stressfields)
        print(".",end="\n")
        self.create_stressfields(component="yy")
        np.save(loc+"/stress_yy.npy",self.stressfields)
        print(f"Done! Saved to : {loc}/stress_ij.npy")
    
    def load_from_npy(self,component="xy"):
        self.stressfields = np.load(self.loc + f"/stress_{component}.npy")
        self.last_component_created=component

        x = np.linspace(0,self.xlength,self.stressfields.shape[1])
        y = np.linspace(0,self.ylength,self.stressfields.shape[2])
        self.mesh_XX,self.mesh_YY = np.meshgrid(x,y)
        self.timesteps = np.linspace(0,self.ntimesteps,self.stressfields.shape[0])
        
        return self.mesh_XX,self.mesh_YY,self.timesteps, self.stressfields
    
    def average_window(self,window_time,at_step=0,component="xy"):
        
        if not hasattr(self,"last_component_created") and self.loadfromnpy:
            self.create_stressfields(component=component)
        elif self.last_component_created != component and not self.loadfromnpy:
            self.create_stressfields(component=component)
        elif self.last_component_created != component and self.loadfromnpy:
            self.load_from_npy(component=component)
        return np.average(self.stressfields[at_step:at_step+window_time,:,:],axis=0)
    
    def average_windows(self,window_time,component="xy"):
        if not hasattr(self,"last_component_created") and not self.loadfromnpy:
            self.create_stressfields(component=component)
        elif self.last_component_created != component and not self.loadfromnpy:
            self.create_stressfields(component=component)
        elif self.last_component_created != component and self.loadfromnpy:
            self.load_from_npy(component=component)
        
        t,x,y = self.stressfields.shape
        self.avg_stressfield = np.zeros((t-window_time,x,y))
        

        for i in range(t-window_time):
            self.avg_stressfield[i,:,:] = self.average_window(window_time,at_step=i,component=component)
        return self.avg_stressfield
    
    def animate(self,window_time,component="xy", cm="bwr",save=None,dpi=100):
        from matplotlib.animation import FuncAnimation
        
        frames = self.average_windows(window_time = window_time,component=component)

        vmax = np.max(np.abs(frames))
        vmin = -vmax

        plt.clf()
        fig = plt.figure()
        im = plt.imshow(frames[0].T,cmap=cm,vmin=vmin,vmax=vmax,extent=[self.mesh_XX[0,0],self.mesh_XX[0,-1],self.mesh_YY[0,0],self.mesh_YY[-1,0]])#aspect="equal",interpolation="none")
        ax = plt.gca()

        def animate_function(i):
            im.set_array(frames[i].T)
            ax.set_title(f"$\\sigma_{{{component}}}$ at t = {i/frames.shape[0]*self.timesteps[-1]*self.dt:.2f} ps")
            return [im]

        anim = FuncAnimation(fig,animate_function,frames=frames.shape[0],interval=100)
        if save:
            anim.save(save, dpi=dpi,extra_args=['-vcodec', 'libx264'])
        plt.show()

    def contourplot_stressfield(self,logplot=False,stress=None,show=False,save=False):
        """
        If you want to save, save should be the string of the path/filename of desired output.
        If stress is None, this plots the stressfield made by the previous create_stressfield call.
        """
        if stress is None and not self.loadfromnpy:
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
        plt.clf()
    
    def avg_tip(self,edge_density,window_time=None,window_space=10,component="xy",cnkargs=(0.004,1,20)):
        """
        Create the average around the cracktip
        edge_density: needs to be an instance of ChunkAvg for finding the cracktip
        window_time: tuple of two timesteps of where to start and end the average
        window_space: the resulting average around the cracktip is this value long in each direction
        component: which component of the stress tensor to average
        """
        
        cracktip_positions = edge_density.find_crack_tips(cnkargs[0],cnkargs[1],ignore_last=cnkargs[2])
        cracktip_timesteps = edge_density.find_global("Timestep")
        

        mesh_XX, mesh_YY, stress_timesteps, stressfields = self.create_stressfields(component=component)
        if window_time is None:
            window_time = (np.min(stress_timesteps),np.max(stress_timesteps))
        
        xaxis = mesh_XX[:,0]

        if cracktip_timesteps.size > stress_timesteps.size:
            factor = int(cracktip_timesteps.size // stress_timesteps.size)
            cracktip_positions = cracktip_positions[::factor]    
            cracktip_timesteps = cracktip_timesteps[::factor]
            indexoffset = int(np.where(cracktip_timesteps==stress_timesteps[0])[0])

        tindices = np.argwhere(np.logical_and(cracktip_timesteps>window_time[0] , cracktip_timesteps<window_time[1]))
        stress_avg = np.zeros((2*window_space,mesh_YY[0,:].size))
        
        for tindex in tindices:
            xindice = np.argmin(np.abs(xaxis-cracktip_positions[tindex]))
            stress_avg += stressfields[tindex[0]-indexoffset,(xindice-window_space):(xindice+window_space),:]
        
        return stress_avg/tindices.size



def _save(direc):
    print(direc)
    stress = Stressfield(direc+"/stressfield.data")
    stress.save_stressfields(loc=direc)

def save_parallel(directory,nproc=8):
    import os
    import multiprocessing as mp

    dirs = [ f.name for f in os.scandir(directory) if f.is_dir() ]
    
    pool = mp.Pool(nproc)
    pool.map(_save,dirs)



def running_average(array,N):
    i,*dims = array.shape

    avg = np.zeros((i-N,*dims))
    for indx in range(i-N):
        avg[indx] = np.average(array[indx:indx+N],axis=0)
    return avg


def alternative_animate(frames,N,title,axis_dims=None, cm="bwr",save=None,dpi=100):
        from matplotlib.animation import FuncAnimation

        frames = running_average(frames,N)        

        vmax = np.max(np.abs(frames))
        vmin = -vmax

        if axis_dims is not None:
            x0,x1,y0,y1 = axis_dims
        else:
            x0,x1,y0,y1 = 0,frames.shape[1],0,frames.shape[2]

        plt.clf()
        fig = plt.figure()
        im = plt.imshow(frames[0].T,cmap=cm,vmin=vmin,vmax=vmax,extent=[x0,x1,y0,y1])#aspect="equal",interpolation="none")
        ax = plt.gca()

        def animate_function(i):
            im.set_array(frames[i].T)
            ax.set_title(f"{title} frame {i}")
            return [im]

        anim = FuncAnimation(fig,animate_function,frames=frames.shape[0],interval=100)
        if save:
            anim.save(save, dpi=dpi,extra_args=['-vcodec', 'libx264'])
        plt.show()