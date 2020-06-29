from lammps_analyzer.stressfield import Stressfield
from lammps_analyzer.chunkavg import ChunkAvg
#from IPython import embed

stress = Stressfield("../../datafiler/ly50_3/stressfield.data")
stress.create_stressfield(component="xy",step=1)
stress.imshow_stressfield(show=True)
