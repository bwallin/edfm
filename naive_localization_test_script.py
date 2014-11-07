'''
Script testing a basic constrained least squares depth-inversion approach for
EDF microscopy.

Author: Bruce Wallin
'''
from __future__ import division
import os
from argparse import ArgumentParser
import logging
import cPickle

from numpy import zeros, array

from psf import DiscretizedPSF
from inversion import LeastSquaresCVXOPT
from misc import load_image


# Generally use the same default template
psf_tiff_dir = '/home/bwallin/ws/edf_micro/data/'
psf_tiff_filename = 'Intensity_PSF_template_CirCau_NA003.tif'
psf_tiff_filepath = os.path.join(psf_tiff_dir, psf_tiff_filename)


def main():
    # Options and arguments
    parser = ArgumentParser(
        description='Perform localization and depth inversion of EDF image.')
    parser.add_argument('image_filepath', metavar='FILEPATH',
                        help='Image to analyze')
    parser.add_argument('-n', '--series-name',
                        dest='series_name', metavar='NAME',
                        default='default_series',
                        help='Series/experiment identifier')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action="store_true", default=False,
                        help='Verbose mode')
    options = parser.parse_args()
    series_name = options.series_name
    image_filepath = options.image_filepath
    formulation = LeastSquaresCVXOPT()
    if options.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # Set up file and console logs
    loglevel = logging.DEBUG
    log_dir = 'logs'
    log_filename = 'naive_localization_l2minl1reg_{}.log'.format(series_name)
    logger = logging.getLogger('EDF naive localization test')
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
    psf_template = DiscretizedPSF()
    psf_template.load_from_tiff(psf_tiff_filepath)
    r, p, q = psf_template.shape
    logger.debug('Done')

    logger.debug('Setting template')
    formulation.set_psf_template(psf_template)
    logger.debug('Done')

    # Data name is meaningful part of filename
    data_name = os.path.splitext(os.path.split(image_filepath)[-1])[0]
    results = {'data_name': data_name,
               'X': []}

    # Load single image tiff
    logger.debug('Loading image: {}'.format(data_name))
    images = load_image(image_filepath)
    for image in images:
        n, m = image.shape
        logger.debug('Done')

        # Initialize depth array
        X = zeros((r, n, m))

        for i in xrange(n):
            for j in xrange(m):
                logger.debug('Setting current image')
                formulation.set_image(image)
                logger.debug('Done')

                logger.debug('Solving')
                formulation.solve(pixel=(i, j))
                x = formulation.x
                X[:, i, j] = array(x[:r]).ravel()
                logger.debug('Done')

        results['X'].append(X)

    cPickle.dump(results, open(series_name+'.pkl', 'wb'))


if __name__ == '__main__':
    main()
