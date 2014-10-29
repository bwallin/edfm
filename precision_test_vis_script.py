import re
import cPickle
from argparse import ArgumentParser

from pylab import *

def main():
    # Options and arguments
    parser = ArgumentParser(description='Analyze and visualize results.')
    parser.add_argument('results_filepath', metavar='filename',
                        help='Results to analyze')
    options = parser.parse_args()

    results = cPickle.load(open(options.results_filepath, 'rb'))

    for depth in ('depth60', 'depth120', 'depth180', 'depth240', 'depth300'):
        data_names = sorted(filter(lambda x: depth in x, results.keys()))
        success_image = zeros((len(data_names), 9))
        for i,data_name in enumerate(data_names):
            # Extract parameters from data name
            true_depth, true_photons, true_G = [float(el) for el in
                    re.search('depth([0-9]*)_p([0-9]*e[0-9]*)_G([0-9]*)', data_name).groups()]
            for j in xrange(9):
                estimated_depth = argmax(results[data_name][j]['result']['x'][:329]) + 1
                error = abs(true_depth - estimated_depth)
                success_image[i, j] = error

        fig, ax = subplots(1,1)
        cs = ax.pcolormesh(success_image)
        cs.set_clim(vmin=0,vmax=35)
        xticks(arange(9)+.5, range(9))
        yticks(arange(len(data_names))+.5, data_names)
        ax.set_xlim([0, 9])
        ax.set_ylim([0, 15])
        ax.set_xlabel('Replication number')
        ax.set_ylabel('Dataset')
        ax.set_title('Estimated depth accuracy')
        colorbar(cs)
        tight_layout()
        savefig('{}_accuracy'.format(depth))

    #for data_name in results.keys():
    #    fig, ax_list = subplots(3, 3, figsize=(12, 9))
    #    for i, ax in enumerate(ax_list.flatten()):
    #        ax.plot(results[data_name][i]['x'][:329])
    #        ax.set_xlim((0, 329))
    #        ax.set_ylim((0, max(results[data_name][i]['x'][:329])))
    #        ax.set_xlabel('Depth index')
    #        ax.set_ylabel('Coefficient')
    #        ax.set_title('Slice {}'.format(i))
    #    tight_layout()
    #    savefig('{}_solutions'.format(data_name))

if __name__ == '__main__': main()
