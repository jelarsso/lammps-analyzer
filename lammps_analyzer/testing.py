from lammps_analyzer.stressfield import Stressfield
from lammps_analyzer.chunkavg import ChunkAvg
from IPython import embed
import matplotlib.pyplot as plt

stress = Stressfield("../../stretch75/stressfield.data")
#stress.create_stressfield(component="xy",step=1)
#stress.imshow_stressfield(show=True)
cnk = ChunkAvg("../../stretch75/edge_density.data")

plt.plot(cnk.find_global("Timestep"),cnk.find_crack_tips(0.004,1,20))
plt.show()

field = stress.avg_tip(cnk,window_time=(5500,45000),window_space=20)

plt.imshow(field.T,cmap="bwr")
plt.title("$\\sigma_{xy}$")
plt.show()



field = stress.avg_tip(cnk,window_time=(5500,45000),window_space=20, component="xx")

plt.imshow(field.T,cmap="bwr")
plt.title("$\\sigma_{xx}$")
plt.show()


field = stress.avg_tip(cnk,window_time=(5500,45000),window_space=20 , component="yy")

plt.imshow(field.T,cmap="bwr")
plt.title("$\\sigma_{yy}$")
plt.show()