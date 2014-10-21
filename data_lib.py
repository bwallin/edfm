'''
Utility functions for EDF microscopy project.

Author: Bruce Wallin
'''
from __future__ import division
import logging

from numpy import zeros, eye, dot, ones, swapaxes
import cvxopt
import cvxopt.solvers
import tifffile


psf_template_filepath = '/home/bwallin/ws/edf_micro/data/Intensity_PSF_template_CirCau_NA003.tif'


def load_psf_template(psf_template_filepath=psf_template_filepath):
    psf_template_tensor = tifffile.imread(psf_template_filepath).astype('float')
    # Now roll first axis to last so in x,y,z order
    psf_template_tensor = swapaxes(psf_template_tensor, 0, 1)
    psf_template_tensor = swapaxes(psf_template_tensor, 1, 2)

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


