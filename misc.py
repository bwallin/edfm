'''
Utility functions for EDF microscopy project.

Author: Bruce Wallin
'''
from __future__ import division
import logging
import functools
import collections

from numpy import zeros, eye, dot, ones, swapaxes
import cvxopt
import cvxopt.solvers
import tifffile


psf_template_filepath = '/home/bwallin/ws/edf_micro/data/Intensity_PSF_template_CirCau_NA003.tif'


def load_psf_template(psf_template_filepath=psf_template_filepath):
    psf_template_tensor = tifffile.imread(psf_template_filepath)

    return psf_template_tensor


def load_image(image_filepath):
    image = tifffile.imread(image_filepath).astype('float')

    return image


def downsample_array(a, ds):
    '''
    Downsample ndarray by factors in vector ds (same length as dimension of a).
    '''
    from skimage.measure import block_reduce
    a_downsampled = block_reduce(a, ds)

    return a_downsampled


def crop_array_union(arrayA, arrayB, location):
    '''
    Crops first two dimensions of \'array_to_crop\' to fit in shape of \'cropper_array\'
    when aligned at \'location\'.
    '''
    i,j = location
    n,m = arrayA.shape[:2]
    p,q = arrayB.shape[:2]
    assert i > -p
    assert i < n+p
    assert j > -q
    assert j < m+q

    left_cropA = max(i, 0)
    right_cropA = min(i+p, n)
    bottom_cropA = max(j, 0)
    top_cropA = min(j+q, m)

    A_sl_h = slice(left_cropA, right_cropA)
    A_sl_v = slice(bottom_cropA, top_cropA)

    left_cropB = max(-i, 0)
    right_cropB = min(n-i, p)
    bottom_cropB = max(-j, 0)
    top_cropB = min(m-j, q)

    B_sl_h = slice(left_cropB, right_cropB)
    B_sl_v = slice(bottom_cropB, top_cropB)

    arrayA_cropped = arrayA[A_sl_h, A_sl_v]
    arrayB_cropped = arrayB[B_sl_h, B_sl_v]

    return arrayA_cropped, arrayB_cropped


class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)
