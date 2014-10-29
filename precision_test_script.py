'''
Script testing a basic constrained least squares depth-inversion approach.

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

from dpsf import DiscretizedPSF
from inversion_formulations import single_loc_lsq_cvxopt, single_loc_l1_cvxopt
from data_lib import load_psf_template, load_image, downsample_array


def main():
    # Options and arguments
    parser = ArgumentParser(description='Perform depth inversion of EDF point source.')
    parser.add_argument('image_filepaths', metavar='filename(s)', nargs='+',
                        help='Image(s) to analyze')
    parser.add_argument('-n', '--series-name', dest='series_name', metavar='NAME',
                        default='default_series', help='Series/experiment identifier')
    parser.add_argument('-o', '--objective-norm', dest='objective_norm', metavar='NAME',
                        default='l2', help='Objective norm on errors to use (l1, l2)')
    parser.add_argument('-d', '--downsample-depth', dest='downsample_depth', type=int,
                        default=1,
                        help='Downsample depth (to avoid poor conditioning, or speed up)')
    parser.add_argument('-D', '--downsample-image', dest='downsample_image', type=int,
                        default=1,
                        help='Downsample image (to speed up solution or reduce memory consumption)')
    parser.add_argument('-c', '--cross-section', dest='cross_section',
                        default=False, action='store_true',
                        help='Subset image and PSF for efficiency (i.e. cross-section, border-crop)')
    parser.add_argument('-g', '--visualize', dest='visualize',
                        action="store_true", default=False,
                        help='Show interactive plots')
    parser.add_argument('-p', '--save-plots', dest='save_plots',
                        action="store_true", default=False,
                        help='Save plots to series name directory')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action="store_true", default=False,
                        help='Verbose mode')
    options = parser.parse_args()
    series_name = options.series_name
    image_filepaths = options.image_filepaths
    if options.objective_norm == 'l2':
        formulation = single_loc_lsq_cvxopt()
    elif options.objective_norm == 'l1':
        formulation = single_loc_l1_cvxopt()
    ds_pixel, ds_depth = options.downsample_image, options.downsample_depth
    cross_section = options.cross_section
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
    psf_template = DiscretizedPSF(depth_resolution=1, depth_unit='?')
    psf_template.load_from_tiff(psf_tiff_path)
    r,p,q = psf_template.shape
    depth_range = arange(1, r+1)*psf_template.depth_resolution
    logger.debug('Shape: {}'.format(psf_template.shape))
    logger.debug('Done')

    logger.debug('Setting template')
    formulation.set_psf_template(psf_template)
    logger.debug('Shape: {}'.format(psf_template.shape))
    logger.debug('Done')

    # Loop over images given on command line
    results = {}
    for image_filepath in image_filepaths:
        # Data name is meaningful part of filename
        data_name = os.path.splitext(os.path.split(image_filepath)[-1])[0]

        # Initialize results dict
        results[data_name] = []

        # Load multipage tiff
        logger.debug('Loading image: {}'.format(data_name))
        images = load_image(image_filepath)
        logger.debug('Shape: {}'.format(images.shape))
        logger.debug('Done')

        for i in xrange(images.shape[0]):
            logger.debug('On slice: {}'.format(i))
            if ds_pixel > 1:
                image = downsample_array(images[i, :, :], (ds_pixel, ds_pixel))
            else:
                image = images[i, :, :]
            n,m = image.shape

            logger.debug('Setting current image')
            if cross_section:
                image = image[:,m/2:m/2+2]
                n,m = image.shape
            formulation.set_image(image)
            logger.debug('Shape: {}'.format(image.shape))
            logger.debug('Done')

            # Initialize results dict entry
            results[data_name].append({})

            logger.debug('Solving')
            start_time = time.time()
            status = formulation.solve()
            solve_time = time.time() - start_time
            logger.debug('Done')

            # Store results
            results[data_name][i]['result'] = formulation.result
            results[data_name][i]['solve_time'] = solve_time

    cPickle.dump(results, open(series_name+'.pkl', 'wb'))

if __name__=='__main__': main()
