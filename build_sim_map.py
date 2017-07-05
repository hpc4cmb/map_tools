#!/usr/bin/env python

# Copyright (c) 2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys
import argparse

import healpy as hp
import numpy as np
from scipy.constants import degree, arcmin, arcsec


def parse_arguments():
    """
    Parse the command line arguments.  Arguments may be entered in a
    parameter file identified from the prefix "@".
    """

    parser = argparse.ArgumentParser(
        description='Synthesize an observed sky map',
        fromfile_prefix_chars='@')

    parser.add_argument('--cmb_scalar', required=False,
                        help='Scalar CMB Alm expansion file')

    parser.add_argument('--cmb_tensor', required=False,
                        help='Tensor CMB Alm expansion file')

    parser.add_argument('--cmb_scale', required=False, type=np.float,
                        help='CMB multiplier to apply')

    parser.add_argument('--cmb_coord', required=False, default='G',
                        help='CMB coordinate system')

    parser.add_argument('--fg', required=False,
                        help='Foreground file')

    parser.add_argument('--fg_scale', required=False, type=np.float,
                        help='Foreground multiplier to apply')

    parser.add_argument('--fg_coord', required=False, default='G',
                        help='Foreground coordinate system')

    parser.add_argument('--lmax', required=False, type=np.int,
                        help='Maximum ell to consider')

    parser.add_argument('--fwhm', required=False, type=np.float,
                        help='Beam to apply [arcmin]')

    parser.add_argument('--nside', required=True, type=np.int,
                        help='Resolution of the output map')

    parser.add_argument('--output_map', required=False,
                        default='total_map.fits.gz',
                        help='Name of the output map')

    parser.add_argument('--output_coord', required=False, default='G',
                        help='Output coordinate system')

    args = parser.parse_args()

    if args.cmb_coord not in 'CEG':
        raise RuntimeError('cmb_coord must be one of [C, E, G]. Not {}'
                           ''.format(args.cmb_coord))

    if args.fg_coord not in 'CEG':
        raise RuntimeError('fg_coord must be one of [C, E, G]. Not {}'
                           ''.format(args.fg_coord))

    if args.output_coord not in 'CEG':
        raise RuntimeError('output_coord must be one of [C, E, G]. Not {}'
                           ''.format(args.output_coord))

    return args


def coordsys2euler_zyz(coord_in, coord_out):
    """
    Return the ZYZ Euler angles corresponding to a rotation between
    the two coordinate systems.  The angles can be used to rotate_alm.
    """

    # Convert basis vectors

    xin, yin, zin = np.eye(3)
    rot = hp.rotator.Rotator(coord=[coord_in, coord_out])
    xout, yout, zout = rot([xin, yin, zin]).T

    # Normalize

    xout /= np.dot(xout, xout)**.5
    yout /= np.dot(yout, yout)**.5
    zout /= np.dot(zout, zout)**.5

    # Get the angles

    psi = np.arctan2(yout[2], -xout[2])
    theta = np.arccos(np.dot(zin, zout))
    phi = np.arctan2(zout[1], zout[0])

    return psi, theta, phi


def read_alm(fn, zero_dipole=False):
    """
    Read polarized alm expansion and return the expansion and lmax.
    Enforce that the expansions are full rank (lmax == mmax).
    """

    alms = []
    lmax = None
    print('Reading', fn)
    for hdu in [1, 2, 3]:
        alm, mmax = hp.read_alm(fn, hdu=hdu, return_mmax=True)
        lmax_temp = hp.Alm.getlmax(alm.size, mmax=mmax)
        if mmax != lmax_temp:
            raise RuntimeError('mmax != lmax in {}[{}]'.format(fn, hdu))
        if lmax is None:
            lmax = lmax_temp
        else:
            if lmax != lmax_temp:
                raise RuntimeError('Alm expansions have different orders')
        alms.append(alm)

    alms = np.array(alms)

    if zero_dipole:
        for ell in range(2):
            for m in range(ell+1):
                i = hp.Alm.getidx(lmax, ell, m)
                alms[0][i] = 0

    return alms, lmax


def read_cmb(args):
    """
    Read (and co-add) the CMB alm expansion.
    """

    # Read scalar CMB

    cmb_scalar = None
    lmax_scalar = -1
    if args.cmb_scalar:
        fn = args.cmb_scalar
        if not os.path.isfile(fn):
            raise RuntimeError('Scalar CMB file not found: {}'.format(fn))
        cmb_scalar, lmax_scalar = read_alm(fn, zero_dipole=True)
        if args.cmb_scale:
            cmb_scalar *= args.cmb_scale

    # Read tensor CMB

    cmb_tensor = None
    lmax_tensor = -1
    if args.cmb_tensor:
        fn = args.cmb_tensor
        if not os.path.isfile(fn):
            raise RuntimeError('Tensor CMB file not found: {}'.format(fn))
        cmb_tensor, lmax_tensor = read_alm(fn, zero_dipole=True)
        if args.cmb_scale:
            cmb_tensor *= args.cmb_scale

    # Combine the expansions

    if cmb_scalar is None and cmb_tensor is None:
        # No CMB loaded
        return None, None

    if args.lmax:
        lmax = args.lmax
    else:
        lmax = max(lmax_scalar, lmax_tensor)

    nalm = hp.Alm.getsize(lmax)
    cmb = np.zeros([3, nalm], dtype=np.complex)

    for ell in range(lmax+1):
        for m in range(ell+1):
            i0 = hp.Alm.getidx(lmax, ell, m)
            if ell <= lmax_scalar:
                i1 = hp.Alm.getidx(lmax_scalar, ell, m)
                cmb[:, i0] += cmb_scalar[:, i1]
            if ell <= lmax_tensor:
                i2 = hp.Alm.getidx(lmax_tensor, ell, m)
                cmb[:, i0] += cmb_tensor[:, i2]

    if args.cmb_coord != args.output_coord:
        psi, theta, phi = coordsys2euler_zyz(args.cmb_coord, args.output_coord)
        hp.rotate_alm(cmb, psi, theta, phi)

    return cmb, lmax


def read_foreground(args):
    """
    Read the foreground and expand it in spherical harmonics.
    """

    fn = args.fg
    if not os.path.isfile(fn):
        raise RuntimeError('Foreground file not found: {}'.format(fn))
    fg_map = np.array(hp.read_map(fn, range(3), verbose=False))

    if args.fg_scale:
        fg_map[fg_map != hp.UNSEEN] *= fg_scale

    if args.lmax:
        lmax = args.lmax
    else:
        lmax = 2*nside

    nside = hp.npix2nside(fg_map[0].size)
    fg = hp.map2alm(fg_map, lmax=lmax, pol=True, iter=0)
    fg = np.array(fg)

    # Invert and deconvolve the pixel window function

    pixwin = np.array(hp.pixwin(nside, pol=True))
    pixwin[pixwin != 0] = 1 / pixwin[pixwin != 0]

    fg[0] = hp.almxfl(fg[0], pixwin[0], inplace=False)
    fg[1] = hp.almxfl(fg[1], pixwin[1], inplace=False)
    fg[2] = hp.almxfl(fg[2], pixwin[1], inplace=False)

    if args.fg_coord != args.output_coord:
        psi, theta, phi = coordsys2euler_zyz(args.fg_coord, args.output_coord)
        hp.rotate_alm(fg, psi, theta, phi)

    return fg, lmax


def smooth_signal(args, cmb_alm, cmb_lmax, fg_alm, fg_lmax):
    """
    Apply beam smoothing to the CMB and foreground expansions and return
    a co-added expansion.
    """

    if args.lmax:
        lmax = args.lmax
    else:
        lmax = max(cmb_lmax, fg_lmax)

    nalm = hp.Alm.getsize(lmax)
    alm = np.zeros([3, nalm], dtype=np.complex)

    for ell in range(lmax+1):
        for m in range(ell+1):
            i0 = hp.Alm.getidx(lmax, ell, m)
            if ell <= cmb_lmax:
                i1 = hp.Alm.getidx(cmb_lmax, ell, m)
                alm[:, i0] += cmb_alm[:, i1]
            if ell <= fg_lmax:
                i2 = hp.Alm.getidx(fg_lmax, ell, m)
                alm[:, i0] += fg_alm[:, i2]

    total = hp.alm2map(alm, args.nside, pixwin=True, verbose=False,
                       fwhm=args.fwhm*arcmin, pol=True, inplace=True)

    return total


def save_map(args, total_map):
    """
    Save the co-added map as a zip-compressed full sky FITS file.
    """

    fn = args.output_map
    if not fn.endswith('.gz'):
        fn += '.gz'

    try:
        hp.write_map(fn, total_map, coord=args.output_coord)
    except:
        hp.write_map(fn, total_map, coord=args.output_coord, overwrite=True)


def main():

    args = parse_arguments()

    cmb_alm, cmb_lmax = read_cmb(args)

    fg_alm, fg_lmax = read_foreground(args)

    sky_map = smooth_signal(args, cmb_alm, cmb_lmax, fg_alm, fg_lmax)

    #noise_map = simulate_noise(args)

    #total_map = sky_map + noise_map

    save_map(args, sky_map)

    return


if __name__ == '__main__':

    main()
