'''
========================================================
Discretized point-spread-functions(:mod:`edfmicro.dpsf`)
========================================================

.. currentmodule:: edfmicro.dpsf

Module fore working with discretized point-spread-functions (PSFs).

Contents
========

Discretized PSF classes
-----------------------

.. autosummary::
   :toctree: generated/

   DiscretizedPSF - Regular grid discretized PSF

Usage Information
=================

Typically a DiscretizedPSF will be initialized from a dense 3-dimensional
numpy array.

>>> from dpsf import DiscretizedPSF
>>> psf = DiscretizedPSF(array([
        [[0,0],[0,1], [0,0]],
>>>     [[0,1],[1,1],[0,1]],
>>>     [[0,0],[0,1],[0,0]]]),
>>>     resolutions=(200,200,200),
>>>     units=('nm', 'nm', 'nm'))
>>> psf.shape
(3,3,2)
>>> psf.template

'''

import scipy.sparse
import tifffile


class DiscretizedPSF(Object):
    '''This class represents a discretized point-spread-function (PSF)
    sampled on a regular grid.

    Parameters
    ----------
    template: scipy.sparse.coo_matrix
        3 dimensional coo_matrix of numeric data type representing PSF
            evaluated at discrete locations
    resolutions: (float, float, float) or None
        Ordered list of step sizes (dx, dy, dz)
    units: (str, str, str) or None
        Ordered list of units for (x, y, z)
    shape: (int, int, int) or None
        Shape of dense representation

    '''
    def __init__(self, dense_array=None, resolutions=None, units=None):
        '''Initialize with parameters or set manually with methods.

        '''
        self.template = None
        self.shape = None
        if dense_array is not None:
            self.set_from_dense_array(dense_array)
        self.resolutions = resolutions
        self.units = units


    def set_from_dense_array(dense_array):
        '''Set the PSF template from dense array

        Parameters
        ----------
        dense_array: array_like
            3 dimensional array_like
        '''

        self.template= scipy.sparse.coo_matrix(dense_array)


    def to_dense_array(self):
        '''Return as 3 dimensional dense numpy array (likely enormous)

        '''
        return self.template.todense()


    def __len__(self):
        '''Translates the dense 3 dimensional discretized PSF array into
        a scipy.sparse.coomatrix.

        Parameters
        ----------

        Returns
        -------
        coomatrix: coomatrix

        '''
        return len(self.template)


    def flatten(self):
        '''Flattens first two dimension (corresponding to (x,y)).

        Returns
        -------
        xy_indices: array_like
            An array with shape (len(self.template), 2) of indices mapping
            values to un-flattend PSF representation.
        values: array_like
            A length len(self.template) one dimensional array of PSF values

        '''



class MaskedPSF(DiscretizedPSF):
    '''This class represents a discretized point-spread-function (PSF)
    sampled on a regular grid.

    Parameters
    ----------
    template: scipy.sparse.coo_matrix
        3 dimensional coo_matrix of numeric data type representing PSF
            evaluated at discrete locations
    resolutions: (float, float, float) or None
        Ordered list of step sizes (dx, dy, dz)
    units: (str, str, str) or None
        Ordered list of units for (x, y, z)

    '''
    def __init__(self, mask=None, *args, **kwargs):
        super(DiscretizedPSF, self).__init__(*args, **kwargs)
        self.set_mask(mask)

    def set_mask(self, mask):
        self.mask = mask


