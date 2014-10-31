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
>>>     [[0,0,0],[0,1,0],[0,0,0]],
>>>     [[0,1,0],[1,1,1],[0,1,0]],
>>>     [[1,1,1],[1,0,1],[1,1,1]]]))

'''
from scipy import zeros
import scipy.sparse
import tifffile


class DiscretizedPSF(object):
    '''This class represents a discretized point-spread-function (PSF)
    sampled on a regular grid.

    Parameters
    ----------
    template: list of scipy.sparse.dok_matrix
        List of dictionary-of-keys sparse matrices of numeric data type
        representing PSF evaluated at discrete locations
    depth_resolution: float or None
        Depth resolution in terms of depth_unit
    pixel_resolution: float or None
        Lateral resolution of pixels in terms of pixel_unit
    depth_unit: str or None
        Unit used for depth
    pixel_unit: str or None
        Unit used for lateral space

    Notes
    -----
    Has a shape, representing the shape of its dense array representation.
    The first axis corresponds to the depth dimension. The second and
    third axes correspond to x,y pixels. The point for which this is the
    this is the point spread function is located at the center in x,y.
    '''
    def __init__(self, dense_array=None, depth_resolution=None,
                 pixel_resolution=None, depth_unit=None, pixel_unit=None):
        '''Initialize with parameters or set manually.  '''
        self._template = None
        self.shape = None
        if dense_array is not None:
            self.set_from_dense_array(dense_array)
        self.depth_resolution = depth_resolution
        self.pixel_resolution = pixel_resolution
        self.depth_unit = depth_unit
        self.pixel_unit = pixel_unit

    def __len__(self):
        ''' Length is number of discrete depths '''
        return len(self._template)

    def set_from_dense_array(self, dense_array):
        '''Set the PSF template from dense array

        Parameters
        ----------
        dense_array: array_like
            3 dimensional array_like. First axis corresponds to depth.

        '''
        r, p, q = dense_array.shape
        self.shape = (r, p, q)
        self._template = [scipy.sparse.coo_matrix(dense_array[i, :, :])
                          for i in xrange(r)]

    def load_from_tiff(self, filepath):
        '''Load from multipage tiff, where each succesive page is an image
        of the PSF at increasing depths.

        Paramters
        ---------
        filepath: str
            Path to tiff

        '''
        dense_array = tifffile.imread(filepath).astype('float')
        self.set_from_dense_array(dense_array)

    def to_dense(self):
        '''Returns dense representation as array.

        Returns
        -------
        dense_array: array_like
            Array over full psf extant with shape (r, p, q)
        '''
        dense_array = zeros(self.shape)
        for i in xrange(self.__len__()):
            dense_array[i, :, :] = self._template[i].todense()

        return dense_array

    def to_dict(self):
        '''Returns sparse representation as a dictionary.

        Keys are integer length 3 tuples corresponding to and values are
        non-zero pixels.

        Returns
        -------
        psf_dok: dict
            Dictionary of key (i, j, k) with i corresponding to depth
            and j,k the x,y pixel indices, and non-zero pixel value
            at that index.

        Notes
        -----
        '''
        psf_dok = {}
        for i in xrange(self.__len__()):
            section_dok = self._template[i].todok()
            for (j, k), value in section_dok.items():
                indices = (i, j, k)
                psf_dok[indices] = value

        return psf_dok
