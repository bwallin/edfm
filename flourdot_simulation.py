import pdb
import cPickle

from numpy import array, zeros
from numpy.random import randint
import tifffile
import click

from data_lib import load_psf_template
from inversion_formulations import single_loc_lsq_cvxopt


def generate_image(locus_shape, num_dots):
    psf_template = load_psf_template()
    p,q,r = psf_template.shape
    depths = randint(0,r, num_dots)
    locs = [(randint(0, locus_shape[0]), randint(0, locus_shape[1])) for _ in xrange(num_dots)]
    intensities = randint(50, 150, num_dots)

    image_shape = array((p,q))+array(locus_shape)
    image = zeros(image_shape)
    for l in xrange(num_dots):
        i,j = locs[l]
        k = depths[l]
        image[i:i+p, j:j+q] += intensities[l]*psf_template[:,:,k]

    return depths, locs, intensities, image


@click.command()
@click.option('--size', '-s', default=5, help='Width of target window in pixels')
@click.option('--num-dots', '-n', default=3, help='Number of dots in target window')
@click.option('--name', '-o', default='localization_test_image.tif', help='Name of output')
@click.option('--plot', '-p', is_flag=True, help='Show plot')
def main(size, num_dots, name, plot):
    locus_shape = (size, size)
    depths, locs, intensities, image = generate_image(locus_shape, num_dots)
    tifffile.imsave(name+'.tif', image)
    cPickle.dump({
            'depths': depths,
            'locs': locs,
            'intensities': intensities
            },
            open(name+'.pkl', 'w')
            )
    if plot:
        from pylab import pcolormesh, show
        pcolormesh(image)
        show()

if __name__=='__main__': main()
