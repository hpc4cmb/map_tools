#!/usr/bin/env python

# Copyright (c) 2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""This script prints out basic statistics about the provided map.
"""

import os
import sys
import argparse

import astropy.io.fits as pf
import healpy as hp
import numpy as np
from scipy.constants import arcmin


def parse_arguments():
    """Parse the command line
    """

    parser = argparse.ArgumentParser(
        description='Analyze the provided map(s)',
        fromfile_prefix_chars='@')
    parser.add_argument('path',
                        help='Path to a map to analyze.')
    #parser.add_argument('path2', nargs='?',
    #                    default=None, help='Path to a map to subtract.')

    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print('Error: not a valid path: {}'.format(args.path))
        sys.exit()

    #if args.path2 is not None:
    #    if not os.path.isfile(args.path2):
    #        print('Error: not a valid path: {}'.format(args.path2))
    #        sys.exit()

    return args


class MapAnalyzer(object):
    """Object to carry out basic statistical analysis for FITS maps.
    """

    def __init__(self, path):
        """ Instantiate a map analyzer.
        """
        self.hdulist = None
        self.hdulist = pf.open(path)
        self.nhdu = len(self.hdulist) - 1

    def __del__(self):
        """ Destruct the map analyzer.
        """
        if self.hdulist is not None:
            self.hdulist.close()

    def analyze_header(self, ihdu=0):
        """Analyze header data
        """

        hdu = self.hdulist[ihdu+1]
        print('HDU = {}'.format(ihdu+1))
        for key in ['nside', 'coordsys', 'ordering']:
            if key in hdu.header:
                value = hdu.header[key]
            else:
                value = 'UNSET'
            print('{:>8} = {}'.format(key, value))

        if 'nside' in hdu.header:
            nside = hdu.header['nside']
            npix = hp.nside2npix(nside)
            wpix = np.sqrt(hp.nside2pixarea(nside)) / arcmin
            print('{:>8} = {}'.format('npix', npix))
            print('{:>8} = {:.3f} arcmin'.format('wpix', wpix))

        keys = ['TYPE', 'FORM', 'UNIT']
        ncol = hdu.header['tfields']

        print('{:8} '.format(''), end='')
        for icol in range(ncol):
            print(' {:15}'.format(icol+1), end='')
        print('')

        for key in keys:
            print('{:>8} '.format(key), end='')
            for icol in range(ncol):
                full_key = 'T' + key + str(icol+1)
                if full_key in hdu.header:
                    value = hdu.header[full_key]
                else:
                    value = 'UNSET'
                print(' {:>15}'.format(value), end='')
            print('')

        return ncol

    def analyze_maps(self, ihdu=0):
        """Analyze actual column data.
        """
        hdu = self.hdulist[ihdu+1]
        stats = ['skyfrac', 'std', 'min', 'max', 'mean']
        for stat in stats:
            print('{:>8} '.format(stat), end='')
            for icol in range(len(hdu.columns)):
                col = hdu.data.field(icol)
                good = np.logical_and(col != 0, col != hp.UNSEEN)
                if stat == 'skyfrac':
                    value = np.sum(good) / col.size
                elif stat == 'std':
                    value = np.std(col[good])
                elif stat == 'min':
                    value = np.amin(col[good])
                elif stat == 'max':
                    value = np.amax(col[good])
                elif stat == 'mean':
                    value = np.mean(col[good])
                print(' {:15.8g}'.format(value), end='')
            print('')


def main():
    """main is executed when the script is run independently.
    """

    args = parse_arguments()

    analyzer = MapAnalyzer(args.path)

    for ihdu in range(analyzer.nhdu):
        analyzer.analyze_header(ihdu)
        analyzer.analyze_maps(ihdu)


if __name__ == '__main__':

    main()
