def labels(thermo):
    '''
    Returns the label associated with thermo.
    
    Supported thermos:  step, elapsed, elaplong, dt, time, cpu, tpcpu, spcpu, 
                        cpuremain, part, timeremain, atoms, temp, press, pe, ke, 
                        etotal, enthalpy, evdwl, ecoul, epair, ebond, eangle, 
                        edihed, eimp, emol, elong, etail, vol, density, lx, ly, 
                        lz, xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz, xlat, 
                        ylat, zlat
    
    :param thermo: thermo style, see https://lammps.sandia.gov/doc/thermo_style.html
    :type thermo: str
    :return: string containing unit
    '''
    
    unit_labels = {"Mass" : "Mass",
                   "Step" : "Step",
                   "Time" : "Time",
                   "dt"   : "Timestep",
                   "elapsed" : "Elapsed time",
                   "elaplong" : "Elapsed time",
                   "cpu" : "CPU time",
                   "tpcpu" : "CPU time",
                   "spcpu" : "CPU time",
                   "cpuremain" : "CPU time remaining",
                   "part" : "Partition",
                   "timeremain" : "Time remaining",
                   "atoms" : "# Atoms",
                   "Temp" : "Temperature",
                   "Press" : "Pressure",
                   "pe" : "Potential energy",
                   "ke" : "Kinetic energy",
                   "TotEng" : "Total energy",
                   "Enthalpy" : "Enthalpy",
                   "evdwl" : "vdW energy",
                   "ecoul" : "Coulomb energy",
                   "epair" : "Pairwise energy",
                   "ebond" : "Bond energy",
                   "eangle" : "Angle energy",
                   "edihed" : "Dihedral energy",
                   "eimp" : "Improper energy",
                   "emol" : "Molecular energy",
                   "elong" : "Long-range energy",
                   "etail" : "Tail energy",
                   "Vol" : "Volume",
                   "Density" : "Density",
                   "lx" : "Length",
                   "ly" : "Length",
                   "lz" : "Length",
                   "xlo" : "x-low",
                   "xhi" : "x-high",
                   "ylo" : "y-low",
                   "yhi" : "y-high",
                   "zlo" : "z-low",
                   "zhi" : "z-high",
                   "xy" : "xy box tilt",
                   "xz" : "xz box tilt",
                   "yz" : "yz box tilt",
                   "xlat" : "Lattice spacing",
                   "ylat" : "Lattice spacing",
                   "zlat" : "Lattice spacing"}
             
    return unit_labels[thermo]

def units(style, thermo):
    '''
    Returns the metal unit associated with thermo.
    
    Supported styles:   metal
    Supported thermos:  step, elapsed, elaplong, dt, time, cpu, tpcpu, spcpu, 
                        cpuremain, part, timeremain, atoms, temp, press, pe, ke, 
                        etotal, enthalpy, evdwl, ecoul, epair, ebond, eangle, 
                        edihed, eimp, emol, elong, etail, vol, density, lx, ly, 
                        lz, xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz, xlat, 
                        ylat, zlat
    
    :param style: unit style, see https://lammps.sandia.gov/doc/units.html
    :type style: str
    :param thermo: thermo style, see https://lammps.sandia.gov/doc/thermo_style.html
    :type thermo: str
    :return: string containing unit
    '''
    
    unit_labels = {"metal" : {
             "Mass" : "g/mol",
             "Step" : "unitless",
             "Time" : "ps",
             "dt" : "ps",
             "elapsed" : "unitless",
             "elaplong" : "unitless",
             "cpu" : "s",
             "tpcpu" : "s",
             "spcpu" : "s",
             "cpuremain" : "s",
             "part" : "unitless",
             "timeremain" : "s",
             "atoms" : "unitless",
             "Temp" : "K",
             "Press" : "Bar",
             "pe" : "eV",
             "ke" : "eV",
             "TotEng" : "eV",
             "Enthalpy" : "eV",
             "evdwl" : "eV",
             "ecoul" : "eV",
             "epair" : "eV",
             "ebond" : "eV",
             "eangle" : "eV",
             "edihed" : "eV",
             "eimp" : "eV",
             "emol" : "eV",
             "elong" : "eV",
             "etail" : "eV",
             "Vol" : "Å^3",
             "Density" : "g/cm^3",
             "lx" : "Å",
             "ly" : "Å",
             "lz" : "Å",
             "xlo" : "Å",
             "xhi" : "Å",
             "ylo" : "Å",
             "yhi" : "Å",
             "zlo" : "Å",
             "zhi" : "Å",
             "xy" : "Å",
             "xz" : "Å",
             "yz" : "Å",
             "xlat" : "Å",
             "ylat" : "Å",
             "zlat" : "Å"}}
             
    return unit_labels[style][thermo]

