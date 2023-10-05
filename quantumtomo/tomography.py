import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from .tomohelpers import *
from .rhoproperties import *


"""
class Rho()

a class defining the resulting density matrix of a tomography prodcedure, holding both the matrix and its key properites.

"""
class Rho():

    def __init__(self, n, d, rho):
        self.dims = [n,n]
        self.d = d
        self.n = n
        self.matrix = rho
        self.purity = purity(self.matrix)
        if self.dims[0] == 2:
            self.s_param = s_param(self.n, self.matrix)
        self.concurrence = concurrence(self.matrix)
        self.tangle = tangle(self.concurrence)
        self.stokes = stokes_params(self.matrix, self.d, self.n)

    def display_stokes(self):
        gell_manns = generate_gellman(self.d)
        sp_matricies = gen_sp_matricies(self.n, gell_manns, gell_manns)
        for i in range(len(sp_matricies)):
            print("Matrix:\n", sp_matricies[i], ". \nAssociated Stokes Parameter:", round(self.stokes[i], 3), "\n\n")

    
"""
class QubitTomo()

object used to perform tomography on qubits.

"""

class QubitTomo():

    def __init__(self, nqbits):
        self.n = nqbits


    """
    qst_MLE(self, projections, counts, filename)

    Performs the Maximum Likelihood Technique Algorithm to reconstruct the state of a qubit. Returns a Rho() object.

    Parameters:
    ----------------------
    projections: list
        list of the states your tomography was projected on, in string form. e.g. ["H", "V", "D", "A", "L", "R"]
    
    counts: numpy array
        counts corresponding to the list of projections (e.g. if projections =[HH, HV...RR], your counts should be written in
        the same order)

    filename: string
        name of the the excel (.xlsx) file you wish to import your data from (to see how to format your data, check out
        example.xlsx)
    """

    def qst_MLE(self, projections=[], counts=np.array([]), filename=None):

        if type(counts) != type(np.array([])):
            raise ValueError("Counts must be a numpy array")

        if filename != None:
            projections, counts = import_xl(self.n, filename)

        else:
            projections = str_to_state(projections, self.n)
        
        rho = maximum_liklihood_estimation(self.n, 2, counts, projections)
        return rho



    """
    bloch_visualization(self, rho):

    Used to visualize a one-qubit state on the bloch sphere using QuTiP (only available for one-qubit tomography)

    Parameters:
    ----------------------

    rho: np.array
        matrix representing the density matrix of the state we wish to plot (must be 2x2)

    """

    def bloch_visualization(self, rho):
        if self.n != 1:
            raise ValueError("bloch_visualization is only available for 1 qubit systems")

        else:
            bloch_sphere(rho)


    """
    density_plot(self, rho):

    Used to visualize any n qubit state on a histogram using QuTiP 

    Parameters:
    ----------------------

    rho: np.array
        matrix representing the density matrix of the state we wish to plot (must be 2**n x 2**n)

    """
    def density_plot(self, rho):
        plot_density(self.n, rho)

    """
    display_rho(self,rho):

    Displays a formatted version of the rho matrix 

    Parameters:
    ----------------------

    rho: np.array
        matrix representing the density matrix of the state we wish to plot (must be 2**n x 2**n)

    """

    def display_rho(self,rho):
        return(qt.Qobj(rho))


"""
class QuditTomo()

object used to perform tomography on qudits.

"""

class QuditTomo():


    def __init__(self, n, dim):
        self.n = n
        self.d = dim

    """
    qst_MLE(self, projections, counts)

    Performs the Maximum Likelihood Technique Algorithm to reconstruct the state of a qubit. Returns a Rho() object.

    Parameters:
    ----------------------
    projections: list
        list of the states your tomography was projected on, in string form. e.g. ["H", "V", "D", "A", "L", "R"]
    
    counts: numpy array
        counts corresponding to the list of projections (e.g. if projections =[HH, HV...RR], your counts should be written in
        the same order)

    filename: string
        name of the the excel (.xlsx) file you wish to import your data from (to see how to format your data, check out
        example.xlsx)
    """

    def qst_MLE(self, projections=[], counts=np.array([])):
        if len(projections) != len(counts):
            raise ValueError("len(projections) mst be equal to len(counts):")

        if type(counts) != type(np.array([])):
            raise ValueError("Counts must be a numpy array")

        rho = maximum_liklihood_estimation(self.n, self.d, counts, projections)
        return Rho(self.n, self.d, rho)


    """
    display_rho(self,rho):

    Displays a formatted version of the rho matrix 

    Parameters:
    ----------------------

    rho: np.array
        matrix representing the density matrix of the state we wish to plot (must be 2**n x 2**n)

    """
    def display_rho(self,rho):
        return(qt.Qobj(rho))


    

    