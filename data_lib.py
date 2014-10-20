'''
Utility functions for EDF microscopy project.

Author: Bruce Wallin
'''
from __future__ import division
import logging

from numpy import zeros, eye, dot, ones
import cvxopt
import cvxopt.solvers
import tifffile


psf_template_filepath = '/home/bwallin/ws/edf_micro/data/Intensity_PSF_template_CirCau_NA003.tif'


def load_psf_template(psf_template_filepath=psf_template_filepath):
    psf_template_tensor = tifffile.imread(psf_template_filepath).astype('float')

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


def clip_by_union(imageA, imageB, centerA=None, centerB=None):
    '''
    Aligns given center pixels of each image and crops both to their logical
    union.
    '''
    n,m = imageA.shape
    p,q = imageB.shape
    if centerA is None:
        centerA = (int(n/2), int(m/2))
    if centerB is None:
        centerB = (int(p/2), int(q/2))

    diffAB = centerA - centerB
    sumAB = centerA + centerB

    left_clip_A, left_clip_B = None, None
    right_clip_A, right_clip_B = None, None
    if diffAB[0] >= 0:
        left_clip_A = diffAB[0]
    else:
        left_clip_B = abs(diffAB[0])
    if sumAB[0] >= n:
        right_clip_B = p-sumAB[0]+n
    else:
        right_clip_A = sumAB[0]

    top_clip_A, top_clip_B = None, None
    bottom_clip_A, bottom_clip_B = None, None
    if diffAB[1] >= 0:
        top_clip_A = diffAB[1]
    else:
        top_clip_B = abs(diffAB[1])
    if sumAB[1] >= m:
        bottom_clip_B = p-sumAB[1]+m
    else:
        bottom_clip_A = sumAB[1]

    A_sl_h = slice(left_clip_A, right_clip_A)
    A_sl_v = slice(bottom_clip_A, bottom_clip_A)

    B_sl_h = slice(left_clip_B, right_clip_B)
    B_sl_v = slice(bottom_clip_B, top_clip_B)

    imageA_clipped = [A_sl_h, A_sl_v]
    imageB_clipped = [B_sl_h, B_sl_v]

    return imageA_clipped, imageB_clipped


