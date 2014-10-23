'''
Script testing a basic constrained least squares depth-inversion approach for EDF microscopy.

Author: Bruce Wallin
'''
from __future__ import division
import re
import os
from glob import glob
import pdb
from argparse import ArgumentParser
import logging
import cPickle
import time

from matplotlib import pyplot as plt
from pylab import (pcolormesh, subplots, show, zeros, arange, dot, array, xlim,
        ylim, eye, savefig, ones, close, vstack, hstack)
from matplotlib.colors import LogNorm
from progressbar import ProgressBar

from inversion_formulations import single_loc_lsq_cvxopt, single_loc_lsql1reg_cvxopt
from data_lib import load_psf_template, load_image, downsample_array


def main():
    # Options and arguments
    parser = ArgumentParser(description='Perform localization and depth inversion of EDF image.')
    parser.add_argument('image_filepath', metavar='FILEPATH',
                        help='Image to analyze')
    parser.add_argument('-n', '--series-name', dest='series_name', metavar='NAME',
                        default='default_series', help='Series/experiment identifier')
    parser.add_argument('-a', '--alpha', dest='alpha', metavar='FLOAT', type=float,
                        default=0, help='Regularization parameter')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action="store_true", default=False,
                        help='Verbose mode')
    options = parser.parse_args()
    series_name = options.series_name
    image_filepath = options.image_filepath
    alpha = options.alpha
    formulation = single_loc_lsql1reg_cvxopt()
    if options.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # Set up file and console logs
    loglevel = logging.DEBUG
    log_dir = 'logs'
    log_filename = 'depth_inversion_precision_test_{}.log'.format(series_name)
    logger = logging.getLogger('EDF inversion - LSQ')
    logger.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, log_filename), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Load PSF template
    logger.debug('Loading template')
    psf_template = load_psf_template()
    p,q,r = psf_template.shape
    depth_range = arange(1, r+1) # in units of delta_depth
    logger.debug('Done')

    logger.debug('Setting template')
    formulation.set_psf_template(psf_template)
    logger.debug('Done')

    logger.debug('Setting alpha to {}'.format(alpha))
    formulation.set_alpha(alpha)
    logger.debug('Done')

    # Data name is meaningful part of filename
    data_name = os.path.splitext(os.path.split(image_filepath)[-1])[0]
    results = {'data_name': data_name}

    # Load single image tiff
    logger.debug('Loading image: {}'.format(data_name))
    image = load_image(image_filepath)
    n,m = image.shape
    logger.debug('Done')

    # Initialize depth array
    X = zeros((n-p, m-q, r))

    #for i in xrange(n-p):
    #    for j in xrange(m-q):
    for i in [2]:
        for j in [22]:
            logger.debug('Setting current image')
            formulation.set_image(image[i:i+p, j:j+q])
            logger.debug('Done')

            logger.debug('Solving')
            start_time = time.time()
            x = formulation.solve()
            X[i,j,:] = array(x[:r]).ravel()
            pdb.set_trace()
            solve_times = start_time - time.time()
            logger.debug('Done')

    results['X'] = X
    cPickle.dump(results, open(series_name+'.pkl', 'wb'))
    pdb.set_trace()

if __name__=='__main__': main()
