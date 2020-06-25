from lammps_analyzer.chunkavg import ChunkAvg
import numpy as np
import matplotlib.pyplot as plt

class Stressfield():



    def __init__(self, filename):

        self.data = ChunkAvg(filename)
        self.component_ordering = {"xx":1, "yy":2, "zz":3, "xy":4} #mapping between the index of the tensor array and component

    
    def _get_axes(self, step=0):
        coordx = self.data.find_local("Coord1",step=step)
        coordy = self.data.find_local("Coord2",step=step)

        #Count how many bins in each dim
        i = 0
        for x in coordx:
            if x  == coordx[0]:
                i+=1
            else:
                break
        
        j = 1
        for y in coordy[1:]:
            if y  == coordy[0]:
                break
            else:
                j+=1
        assert i*j == self.data.find_global("Number-of-chunks")[step], "Error in counting bins per axis"
        
        mesh_XX = np.resize(coordx,(i,j))
        mesh_YY = np.resize(coordy,(i,j))

        nbins_X = i
        nbins_Y = j

        return mesh_XX,mesh_YY,nbins_X,nbins_Y


    def create_stressfield(self, step=0, component="xx"):
        
        mesh_XX,mesh_YY,nbins_X,nbins_Y = self._get_axes(step=step)
        component_index = self.component_ordering[component]

        stressfield = self.data.find_local(f"c_peratom[{str(component_index)}]",step=step)
        stressfield = np.resize(stressfield,(nbins_X,nbins_Y))
        
        self.stress = (mesh_XX,mesh_YY,stressfield,nbins_X,nbins_Y,component,step)
    

    def contourplot_stressfield(self,stress=None,show=False,save=False):
        """
        If you want to save, save should be the string of the path/filename of desired output.
        If stress is None, this plots the stressfield made by the previous create_stressfield call.
        """
        if stress is None:
            assert hasattr(self,"stress"), "create_stressfield must be called first or stress field passed as arg!"
            stress = self.stress
        
        mesh_XX,mesh_YY,stressfield,nbins_X,nbins_Y,component,step = stress

        plt.contourf(mesh_XX,mesh_YY,stressfield)
        plt.title(f"Stress field $\\sigma_{{{component}}}$ nbins = {nbins_X*nbins_Y} at timestep = {step}")
        plt.xlabel("x-length [Å]")
        plt.ylabel("y-length [Å]")
        
        if save:
            plt.savefig(save)
        if show:
            plt.show()
        






