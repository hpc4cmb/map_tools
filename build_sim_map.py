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

    parser.add_argument('--debug',
                        required=False, default=False, action='store_true',
                        help='Output debugging information')

    parser.add_argument('--noise_tt_sigma', required=False, type=np.float,
                        help='TT white noise level [uK-arcmin]')

    parser.add_argument('--noise_tt_knee', required=False, type=np.float,
                        help='TT noise knee multipole')

    parser.add_argument('--noise_tt_slope', required=False, type=np.float,
                        help='TT noise slope')

    parser.add_argument('--noise_ee_sigma', required=False, type=np.float,
                        help='EE white noise level [uK-arcmin]')

    parser.add_argument('--noise_ee_knee', required=False, type=np.float,
                        help='EE noise knee multipole')

    parser.add_argument('--noise_ee_slope', required=False, type=np.float,
                        help='EE noise slope')

    parser.add_argument('--noise_bb_sigma', required=False, type=np.float,
                        help='BB white noise level [uK-arcmin]')

    parser.add_argument('--noise_bb_knee', required=False, type=np.float,
                        help='BB noise knee multipole')

    parser.add_argument('--noise_bb_slope', required=False, type=np.float,
                        help='BB noise slope')

    parser.add_argument('--hit', required=False,
                        help='Proportional hitmap')

    parser.add_argument('--hit_coord', required=False, default='G',
                        help='hitmap coordinate system')

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

    parser.add_argument('--lmin', required=False, type=np.int,
                        help='Minimum ell to consider')

    parser.add_argument('--lmax', required=False, type=np.int,
                        help='Maximum ell to consider')

    parser.add_argument('--fwhm', required=False, type=np.float,
                        help='Beam to apply [arcmin]')

    parser.add_argument('--nside', required=True, type=np.int,
                        help='Resolution of the output map')

    parser.add_argument('--output', required=False,
                        default='simulation',
                        help='Root name for all outputs')

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


def read_noise(args):
    """
    Parse the noise parameters into noise spectra and per pixel noise
    levels.  Zero knee multipole produces a white noise spectrum.
    """

    if args.lmax:
        lmax = args.lmax
    else:
        lmax = 2*args.nside

    noise_cl = np.zeros([4, lmax+1]) # TT, EE, BB and TE

    ell = np.arange(lmax+1)
    norm = 1e-6 * arcmin

    # TT noise spectrum

    sigma = args.noise_tt_sigma * norm
    if args.noise_tt_knee:
        knee = 1 / args.noise_tt_knee
    else:
        knee = 0
    slope = args.noise_tt_slope
    if knee:
        noise_cl[0][1:] = (1 + (ell[1:]*knee)**slope) * sigma**2
    else:
        noise_cl[0][1:] = sigma**2
    sigma_tt = sigma

    # EE noise spectrum (use TT parameters when EE are absent)

    if args.noise_ee_sigma:
        sigma = args.noise_ee_sigma * norm
    if args.noise_ee_knee:
        knee = 1 / args.noise_ee_knee
    if args.noise_ee_slope:
        slope = args.noise_ee_slope
    if knee:
        noise_cl[1][2:] = (1 + (ell[2:]*knee)**slope) * sigma**2
    else:
        noise_cl[1][2:] = sigma**2
    sigma_ee = sigma

    # BB noise spectrum (use EE parameters when BB are absent)

    if args.noise_bb_sigma:
        sigma = args.noise_bb_sigma * norm
    if args.noise_bb_knee:
        knee = 1 / args.noise_bb_knee
    if args.noise_bb_slope:
        slope = args.noise_bb_slope
    if knee:
        noise_cl[2][2:] = (1 + (ell[2:]*knee)**slope) * sigma**2
    else:
        noise_cl[2][2:] = sigma**2
    sigma_bb = sigma

    if args.debug:
        fn = args.output + '_noise_cl.fits'
        if os.path.isfile(fn):
            os.remove(fn)
        print('Writing N_ell to', fn)
        hp.write_cl(fn, list(noise_cl))

    # Split the noise into white and 1/f parts.  White noise is best
    # simulated in the pixel domain.

    nside = args.nside
    npix = 12*nside**2
    wpix = np.sqrt(4*np.pi/npix)

    noise_cl[0][1:] -= sigma_tt**2
    sigma_tt /= wpix

    sigma_pol = min(sigma_ee, sigma_bb)
    noise_cl[1][2:] -= sigma_pol**2
    noise_cl[2][2:] -= sigma_pol**2
    sigma_pol /= wpix

    return noise_cl, sigma_tt, sigma_pol


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
        fg_map[fg_map != hp.UNSEEN] *= args.fg_scale

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

    hp.almxfl(fg[0], pixwin[0], inplace=True)
    hp.almxfl(fg[1], pixwin[1], inplace=True)
    hp.almxfl(fg[2], pixwin[1], inplace=True)

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

    signal = hp.alm2map(list(alm), args.nside, pixwin=True, verbose=False,
                        fwhm=args.fwhm*arcmin, pol=True, inplace=True)

    if args.debug:
        fn = args.output + '_signal_map.fits.gz'
        if os.path.isfile(fn):
            os.remove(fn)
        print('Writing signal map to', fn)
        hp.write_map(fn, signal, coord=args.output_coord)

    return signal


def save_map(args, total_map):
    """
    Save the co-added map as a zip-compressed full sky FITS file.
    """

    fn = args.output + '_map.fits.gz'

    if os.path.isfile(fn):
        os.remove(fn)

    hp.write_map(fn, total_map, coord=args.output_coord)

    return


def simulate_noise(args, noise_cl, sigma_tt, sigma_pol):
    """
    Simulate a noise map using the N_ell and white noise levels.
    We will assume diagonal white noise covariance matrices for now.
    """

    if args.lmax:
        lmax = args.lmax
    else:
        lmax = 2*args.nside

    # Simulate correlated noise modes in spherical harmonics

    nalm = hp.Alm.getsize(lmax)
    noise_alm = np.zeros([3, nalm], dtype=np.complex)

    noise_alm[0][1:] = np.random.randn(nalm-1) + np.random.randn(nalm-1)*1j
    noise_alm[1][2:] = np.random.randn(nalm-2) + np.random.randn(nalm-2)*1j
    noise_alm[2][2:] = np.random.randn(nalm-2) + np.random.randn(nalm-2)*1j

    norm = 1 / np.sqrt(2)
    hp.almxfl(noise_alm[0], noise_cl[0]**.5*norm, inplace=True)
    hp.almxfl(noise_alm[1], noise_cl[1]**.5*norm, inplace=True)
    hp.almxfl(noise_alm[2], noise_cl[2]**.5*norm, inplace=True)

    # Transform to pixel domain

    noise_map = np.array(hp.alm2map(list(noise_alm), args.nside, pixwin=False,
                                    verbose=False, pol=True, inplace=True))

    # Add white noise in pixel space

    npix = 12*args.nside**2
    noise_map[0] += np.random.randn(npix)*sigma_tt
    noise_map[1] += np.random.randn(npix)*sigma_pol
    noise_map[2] += np.random.randn(npix)*sigma_pol

    if args.debug:
        fn = args.output + '_noise_map.fits.gz'
        if os.path.isfile(fn):
            os.remove(fn)
        print('Writing noise map to', fn)
        hp.write_map(fn, noise_map, coord=args.output_coord)

    return noise_map


def main():

    args = parse_arguments()

    cmb_alm, cmb_lmax = read_cmb(args)

    fg_alm, fg_lmax = read_foreground(args)

    sky_map = smooth_signal(args, cmb_alm, cmb_lmax, fg_alm, fg_lmax)

    noise_cl, sigma_tt, sigma_pol = read_noise(args)

    noise_map = simulate_noise(args, noise_cl, sigma_tt, sigma_pol)

    total_map = sky_map + noise_map

    save_map(args, sky_map)

    return


if __name__ == '__main__':

    main()
