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

from formulations import single_loc_lsq_cvxopt
from data_lib import load_psf_template, load_image, clip_by_union, downsample_array


def main():
    # Options and arguments
    parser = ArgumentParser(description='Perform depth inversion of EDF point source.')
    parser.add_argument('image_filepaths', metavar='filename(s)', nargs='+',
                        help='Image(s) to analyze')
    parser.add_argument('-n', '--series-name', dest='series_name', metavar='NAME',
                        default='default_series', help='Series/experiment identifier')
    parser.add_argument('-d', '--downsample-depth', dest='downsample_depth', type=int,
                        default=1, help='Downsample depth (to avoid poor conditioning, or speed up)')
    parser.add_argument('-D', '--downsample-image', dest='downsample_image', type=int,
                        default=1, help='Downsample image (to speed up solution or reduce memory consumption)')
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
    formulation = single_loc_lsq_cvxopt()
    ds_pixel, ds_depth = options.downsample_image, options.downsample_depth
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
    psf_template = downsample_array(load_psf_template(), (ds_pixel, ds_pixel, ds_depth))
    r,p,q = psf_template.shape
    depth_range = arange(1, r+1) # in units of delta_depth*ds_depth
    logger.debug('Done')

    logger.debug('Setting template')
    formulation.set_psf_template(psf_template)
    logger.debug('Done')

    # Loop over images given on command line
    results = {}
    for image_filepath in image_filepaths:
        # Data name is meaningful part of filename
        data_name = os.path.splitext(os.path.split(image_filepath)[-1])[0]

        # Extract parameters from data name
        true_depth, true_photons, true_G = [float(el) for el in
                re.search('depth([0-9]*)_p([0-9]*e[0-9]*)_G([0-9]*)', data_name).groups()]

        # Initialize results dict
        results[data_name] = []

        # Load multipage tiff
        logger.debug('Loading image: {}'.format(data_name))
        images = load_image(image_filepath)
        logger.debug('Done')

        for i in xrange(images.shape[0]):
            logger.debug('On slice: {}'.format(i))
            image = downsample_array(images[i, :, :], (ds_pixel, ds_pixel))
            n,m = image.shape

            logger.debug('Setting current image')
            formulation.set_image(image)
            logger.debug('Done')

            # Initialize results dict entry
            results[data_name].append({})

            logger.debug('Solving')
            start_time = time.time()
            x = formulation.solve()
            solve_time = start_time - time.time()
            logger.debug('Done')

            # Compute estimated point source depth, and store results
            depth_mode = array(x)[:r].argmax() + 1
            results[data_name][i]['depth_abserror'] = abs(true_depth - depth_mode)
            results[data_name][i]['x'] = array(x)
            results[data_name][i]['solve_time'] = solve_time
            logger.debug('Depth error: {}'.format(results[data_name][i]['depth_abserror']))

            # Optionally make solution plot
            if options.visualize or options.save_plots:
                fig, ax1 = subplots(1, 1)
                ax1.plot(depth_range, x[:r], '-')
                ax1.vlines(true_depth, 0, max(x), 'g')
                ax1.set_xlim((min(depth_range), max(depth_range)))
                ax1.set_ylim((0, max(x)))
                ax1.set_xlabel('depth')
                ax1.set_ylabel('coefficient')
                ax1.set_title('Solution to constrained least squares: {} #{}'.format(data_name, i))
                plot_dir = os.path.join('.', 'plots', series_name)
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                if options.visualize:
                    logger.debug('Showing plot')
                    show()
                    logger.debug('Done')
                if options.save_plots:
                    plot_filename = os.path.join(plot_dir, data_name+'_'+str(i)+'.png')
                    logger.debug('Saving plot to {}'.format(plot_filename))
                    savefig(plot_filename)
                    logger.debug('Done')
                close(fig)

    cPickle.dump(results, open(series_name+'.pkl', 'wb'))

if __name__=='__main__': main()
