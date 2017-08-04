# Copyright (c) 2017 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

""" This module provides the plug_holes function.
"""

import numpy as np

import healpy as hp


def plug_holes(input_map, verbose=False, in_place=True, nest=False):
    """ Use downgrading to derive estimates of the missing pixel values.

    This method is useful when preparing to smooth or sample a map.

    Args:
        input_map (ndarray): Map to plug.
        verbose (bool): Toggle verbosity.
        in_place (bool): Fill the holes in the input array.
        nest (bool): Input map is in NESTED Healpix ordering.

    Returns:
        ndarray:  Reference to a map with the holes plugged.

    """
    nbad_start = np.sum(np.isclose(input_map, hp.UNSEEN))

    if nbad_start == input_map.size:
        if verbose:
            print('plug_holes: All map pixels are empty. Cannot plug holes',
                  flush=True)
        return

    if nbad_start == 0:
        if verbose:
            print('plug_holes: All map pixels are full. Cannot plug holes',
                  flush=True)
        return

    nside = hp.get_nside(input_map)
    npix = hp.nside2npix(nside)
    if in_place:
        output_map = input_map
    else:
        output_map = np.copy(input_map)
    if not nest:
        output_map[:] = hp.reorder(output_map, r2n=True)

    lowres = output_map
    nside_lowres = nside
    bad = np.isclose(output_map, hp.UNSEEN)
    while np.any(bad) and nside_lowres > 1:
        nside_lowres //= 2
        lowres = hp.ud_grade(lowres, nside_lowres, order_in='NESTED')
        hires = hp.ud_grade(lowres, nside, order_in='NESTED')
        bad = np.isclose(output_map, hp.UNSEEN)
        output_map[bad] = hires[bad]

    nbad_end = np.sum(bad)

    if nbad_end != 0:
        mapmean = np.mean(output_map[np.logical_not(bad)])
        output_map[bad] = mapmean

    if not nest:
        output_map[:] = hp.reorder(output_map, n2r=True)

    if verbose and nbad_start != 0:
        print('plug_holes: Filled {} missing pixels ({:.2f}%), lowest '
              'resolution was Nside={}.'.format(
                  nbad_start, (100.*nbad_start)//npix, nside_lowres))

    return output_map
