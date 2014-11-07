'''
Script testing a basic constrained least squares depth-inversion approach.

Author: Bruce Wallin
'''
from __future__ import division
import os
from argparse import ArgumentParser
import logging
import cPickle
import time

from psf import DiscretizedPSF
from inversion import LeastSquaresCVXOPT
from misc import downsample_array, load_image

psf_tiff_dir = '/home/bwallin/ws/edf_micro/data/'
psf_tiff_filename = 'Intensity_PSF_template_CirCau_NA003.tif'
psf_tiff_filepath = os.path.join(psf_tiff_dir, psf_tiff_filename)


def main():
    # Options and arguments
    parser = ArgumentParser(
        description='Perform depth inversion of EDF point source.')
    parser.add_argument('image_filepaths', metavar='filename(s)', nargs='+',
                        help='Image(s) to analyze')
    parser.add_argument('-n', '--series-name',
                        dest='series_name', metavar='NAME',
                        default='default_series',
                        help='Series/experiment identifier')
    parser.add_argument('-D', '--downsample-image',
                        dest='downsample_image', type=int, default=1,
                        help='Downsample image (to speed up solution or \
                        reduce memory consumption)')
    parser.add_argument('-c', '--cross-section',
                        dest='cross_section', default=False,
                        action='store_true',
                        help='Subset image and PSF for efficiency \
                        (i.e. cross-section, border-crop)')
    parser.add_argument('-g', '--visualize',
                        dest='visualize', action="store_true", default=False,
                        help='Show interactive plots')
    parser.add_argument('-p', '--save-plots',
                        dest='save_plots', action="store_true", default=False,
                        help='Save plots to series name directory')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action="store_true", default=False,
                        help='Verbose mode')
    options = parser.parse_args()
    series_name = options.series_name
    image_filepaths = options.image_filepaths
    formulation = LeastSquaresCVXOPT()
    ds_pixel = options.downsample_image
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
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    psf_template.load_from_tiff(psf_tiff_filepath)
    r, p, q = psf_template.shape
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
            n, m = image.shape

            logger.debug('Setting current image')
            if cross_section:
                image = image[:, m/2:m/2+1]
                n, m = image.shape
            formulation.set_image(image)
            logger.debug('Shape: {}'.format(image.shape))
            logger.debug('Done')

            # Initialize results dict entry
            results[data_name].append({})

            logger.debug('Solving')
            start_time = time.time()
            formulation.solve()
            solve_time = time.time() - start_time
            logger.debug('Done')

            # Store results
            results[data_name][i]['result'] = formulation.result
            results[data_name][i]['x'] = formulation.x
            results[data_name][i]['solve_time'] = solve_time

    cPickle.dump(results, open(series_name+'.pkl', 'wb'))

if __name__ == '__main__':
    main()
