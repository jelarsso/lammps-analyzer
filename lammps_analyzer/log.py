import numpy as np

class Log:
    """ Analyzing log files, containing system information.
    
    Parameters
    ----------
    filename : str
        string containing file to load.
    ignore_first : int
        ignore equilibriation files.
    """
    def __init__(self, filename, ignore_first=0):
        self.read_log_file(filename, ignore_first)
        self.mk_dumpfile = True

    def read_log_file(self, filename, ignore_first):
        """ Reading log file by going through file line after line. 
        
        Parameters
        ----------
        filename : str
            string containing file to load.
        ignore_first : int
            ignore equilibriation files.
        """
        f = open(filename, "r")

        self.timestep = 0.005        # Default
        self.mass = 1                # Default
        self.units = "metal"         # Default
        self.lst = []

        read = False                # True if the line should be read
        for i, line in enumerate(f.readlines()):
            # Search for variables
            if line.startswith("Step"):
                self.variables = line.split()
                num_variables = len(self.variables)
                self.lst.append([[] for _ in range(num_variables)])
                read = True
            elif line.startswith("Loop time of"):
                read = False
            elif read:
                strings = line.split()
                for j, string in enumerate(strings):
                    self.lst[-1][j].append(float(line.split()[j]))
            # Search for timestep
            elif line.startswith("timestep"):
                self.timestep = line.split()[1]
            # Search for mass
            elif line.startswith("mass"):
                self.mass = float(line.split()[1])
            elif line.startswith("units"):
                self.units = line.split()[1]
        f.close()
        self.array = np.hstack(self.lst[ignore_first:])
        self.shape = self.array.shape
        
                
    def find(self, key):
        """ Search for a category (Step, Temp, Press etc...). If the 
        keyword exists, if returns the associated array containing
        the quantity as a function of timesteps.
        
        Parameters
        ----------
        key : str
            string containing keyword
            
        Returns
        -------
        ndarray
            array containing the column related to the key
        """
        array = None
        for i, variable in enumerate(self.variables):
            if variable == key:
                array = self.array[i]
        if array is None:
            raise KeyError("No category named {} found.".format(key))
        else:
            return np.array(array)
        
    def smooth(self, y, window_size=29, order=4, deriv=0, rate=1):
        """ Smooth data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        
        Parameters
        ----------
        y : array_like, shape(N,) 
            the values of the time history of the signal
        window_size : int
            the length of the window. Must be an odd integer number
        order : int 
            the order of the polynomial used in the filtering. 
            Must be less then `window_size` - 1.
        deriv : int
            the order of the derivative to compute (default = 0 means only smoothing)

        Returns
        -------
        ndarray
            the smoothed signal (or it's n-th derivative)
        
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
        import scipy.signal

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * np.math.factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')
        
    def plot(self, x, y, show=False, save=False, smooth=False, window_size=501, 
             order=4, xlabel=None, ylabel=None, xunit=None, yunit=None):
        """ Plot a column x as a function of column y using matplotlib. Note 
        that both x and y are supposed to be the names of the observables that
        we want to plot, and are therefore strings. 

        Parameters
        ----------
        x : str
            string containing which column to plot on the x-axis
        y : str
            string containing which column to plot on the y-axis
        show : bool or int 
            show plot
        save : str or bool or list 
            where to save plot. If nothing is specified, the plot is not saved. 
            If save=True, the image is saved in the current directory. 
            If save is string of filenames, the image is saved as all filenames. 
        smooth : bool or int 
            smooth curve using the Savitzky-Golay filter
        window_size : int 
            window size of the Savitzky-Golay filter. Has impact only if smooth=True.
        order : int 
            order of polynomial smoothing by the Savitzky-Golay filter. 
            Has impact only if smooth=True.
        xlabel : str 
            label on x-axis
        ylabel : str 
            label on y-axis
        xunit : str
            unit on x-axis
        yunit : str
            unit on y-axis
            
        Returns
        -------
        list On Python 3.x, this returns a dictionary view object, not a list
        """
        x_array = self.find(x)       # x-array
        y_array = self.find(y)       # y-array
        
        from labels import labels, units
        if xlabel is None:
            try:
                xlabel = labels(x)
            except NameError:
                print("Unknown x-label")
                xlabel = "unknown"
        if ylabel is None:
            try:
                ylabel = labels(y)
            except NameError:
                print("Unknown y-label")
                ylabel = "unknown"
        if xunit is None:
            try:
                xunit = units("metal", x)
            except NameError:
                print("Unknown x-unit")
                xunit = "unknown"
        if yunit is None:
            try:
                yunit = units("metal", y)
            except NameError:
                print("Unknown y-unit")
                yunit = "unknown"
        
        if smooth: 
            x_array = self.smooth(x_array, window_size, order)
            y_array = self.smooth(y_array, window_size, order)
        
        import matplotlib.pyplot as plt
        plt.style.use("bmh")
        plt.rcParams["font.family"] = "Serif"   # Font
        plt.rcParams.update({'figure.autolayout': True})  # Autolayout
        plt.rc('figure', max_open_warning = 0)      # Avoiding RecursionError
        plt.figure()
        plt.plot(x_array, y_array)
        plt.xlabel(xlabel + " [" + xunit + "]")
        plt.ylabel(ylabel + " [" + yunit + "]")
        if type(save) is str: 
            plt.savefig(save)
        elif type(save) is list:
            for filename in save:
                plt.savefig(filename)
        elif save:
            plt.savefig("{}_{}.png".format(x, y))
        if show: 
            plt.show()
        return None
        
    def estimate_boiling_temperature(self, units='K', prnt=False, dumpfile=None):
        """ Estimates the boiling temperature from the graph of the total
        energy. The water is boiling where the energy has the highest slope.
        
        Supported units: K, C, F
        
        Parameters
        ----------
        units : str
            in which temperature units the boiling temperature is displayed and stored
        prnt : bool or int
            printing the boiling temperature to the terminal is prnt=True
        
        Returns
        -------
        float
            estimated boiling temperature
        """
        temp = self.find("Temp")
        energy = self.find("TotEng")
        energy = self.smooth(energy)
        energy_der = self.smooth(energy, deriv=1)
        index = np.argmax(energy_der)
        
        if units == "C":
            temp -= 273.15
        elif units == "F":
            temp = temp * 9/5. - 459.67
        
        T_boil = temp[index]
        if prnt: print("Estimated boiling temperature is {} {}".format(T_boil, units))
        if type(dumpfile) is str:
            arg = 'a'
            if self.mk_dumpfile:
                arg = 'w'
            with open(dumpfile, arg) as file:
                file.write("# Estimated boiling temperature given in {}:\n".format(units))
                file.write(str(T_boil) + "\n\n")
                file.write("")
            self.mk_dumpfile = False
        return T_boil
        
    def estimate_vaporization_enthalpy(self, units='eV', atoms=None, prnt=False, dumpfile=None):
        """ Estimates the vaporization enthalpy based on the enthalpy 
        change around boiling point.
        
        Supported units: eV, kcal, kcal/mol
        
        Parameters
        ----------
        units : str
            in which temperature units the vaporization enthalpy is displayed 
            and stored
        prnt : bool or int
            printing the vaporization enthalpy to the terminal is prnt=True
        
        Returns
        -------
        float
            estimated vaporization enthalpy
        """
        enthalpy = self.find("Enthalpy")
        enthalpy = self.smooth(enthalpy)
        
        if units == "kcal":
            enthalpy /= 2.61144742e22
        elif units == "kcal/mol":
            NA = 6.02214075e23          # Avogadro's number, mol^-1
            mol = atoms / NA
            enthalpy /= 2.61144742e22 * mol
        H_boil = np.max(enthalpy) - np.min(enthalpy)
        if prnt: print("Estimated vaporization enthalpy is {} {}".format(H_boil, units))
        if type(dumpfile) is str:
            arg = 'a'
            if self.mk_dumpfile:
                arg = 'w'
            with open(dumpfile, arg) as file:
                file.write("# Estimated vaporization enthalpy given in {}:\n".format(units))
                file.write(str(H_boil) + "\n\n")
                file.write("")
            self.mk_dumpfile = False
        return H_boil
        
if __name__ == "__main__":
    # EXAMPLE USAGE
    logger = Log("argon_nve_0.005.log")
    
    # Keys available
    print(logger.variables)
    
    # Plot energy
    #logger.plot("Time", "v_msd", show=True)
    
    time = logger.find("Time")
    msd = logger.find("v_msd")
    
    import matplotlib.pyplot as plt
    plt.style.use("bmh")
    plt.rcParams["font.family"] = "Serif"   # Font
    plt.rcParams.update({'figure.autolayout': True})  # Autolayout
    plt.plot(time, msd)
    plt.show()
