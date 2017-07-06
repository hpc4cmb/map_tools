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


def parse_arguments(header):
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

    parser.add_argument('--freq', required=False, type=np.int,
                        default=0, help='Observing frequency [GHz]')

    parser.add_argument('--seed', required=False, type=np.int,
                        default=123456, help='Random number seed seed')

    parser.add_argument('--realization', required=False, type=np.int,
                        default=0, help='CMB and/or noise realization')

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

    parser.add_argument('--noise_scale', required=False, type=np.float,
                        help='Noise multiplier to apply')

    parser.add_argument('--hit', required=False,
                        help='Proportional hitmap')

    parser.add_argument('--hit_coord', required=False, default='G',
                        help='hitmap coordinate system')

    parser.add_argument('--cmb', required=False, action='append',
                        help='CMB Alm expansion file '
                        '(can be defined multiple times)')

    parser.add_argument('--cmb_scale', required=False, type=np.float,
                        help='CMB multiplier to apply')

    parser.add_argument('--cmb_coord', required=False, default='G',
                        help='CMB coordinate system')

    parser.add_argument('--fg', required=False, action='append',
                        help='Foreground file '
                        '(can be defined multiple times)')

    parser.add_argument('--fg_scale', required=False, type=np.float,
                        help='Foreground multiplier to apply')

    parser.add_argument('--fg_coord', required=False, default='G',
                        help='Foreground coordinate system')

    parser.add_argument('--lmin', required=False, type=np.int,
                        help='Highpass filter scale')

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

    print('\nAll parameters:\n', args, '\n')

    header.append(('freq', args.freq, 'Frequency [GHz]'))
    header.append(('real', args.realization, 'Realization'))

    return args


def apply_hits(args, sky_map, noise_map, header):

    if not args.hit:
        return sky_map, noise_map

    print('Applying hit map')

    hitheader = []

    fn = args.hit
    if not os.path.isfile(fn):
        raise RuntimeError('Hit file not found: {}'.format(fn))
    print('  Reading', fn)
    hit = hp.read_map(fn, dtype=np.float)
    hit[hit == hp.UNSEEN] = 0
    nnan = np.sum(np.isnan(hit))
    if nnan != 0:
        print('WARNING: setting {} NANs in {} to zero.'.format(nnan, fn))
        hit[np.isnan(hit)] = 0
    hitheader.append(('hit', fn, 'Relative hit map'))

    nside_hit = hp.npix2nside(hit.size)
    hitheader.append(('hitnside', nside_hit, 'Relative hit map Nside'))
    hitheader.append(('hitcoord', args.hit_coord, 'Relative hit map coord'))

    modified = False
    if args.hit_coord != args.output_coord:
        print('  Rotating {} -> {}'.format(args.hit_coord, args.output_coord))
        lmax_hit = 2*nside_hit
        hit_alm = hp.map2alm(hit, lmax=lmax_hit, iter=0)
        psi, theta, phi = coordsys2euler_zyz(args.hit_coord, args.output_coord)
        hp.rotate_alm(hit_alm, psi, theta, phi)
        hit = hp.alm2map(hit_alm, args.nside, pixwin=False, verbose=False)
        hit[hit < 1e-3] = 0
        modified = True
    elif nside_hit != args.nside:
        print('  UDgrading {} -> {}', nside_hit, args.nside)
        hit = hp.ud_grade(hit, args.nside)
        modified = True

    if args.debug and modified:
        fn = args.output + '_hit_map.fits.gz'
        if os.path.isfile(fn):
            os.remove(fn)
        print('  Writing hit map to', fn)
        hp.write_map(fn, hit, coord=args.output_coord, extra_header=hitheader)

    if np.atleast_1d(sky_map).size != 1:
        unseen = hit == 0
        sky_map[0][unseen] = hp.UNSEEN
        sky_map[1][unseen] = hp.UNSEEN
        sky_map[2][unseen] = hp.UNSEEN

    if np.atleast_1d(noise_map).size != 1:
        seen = hit != 0
        scale = 1 / np.sqrt(hit[seen])
        noise_map[0][seen] *= scale
        noise_map[1][seen] *= scale
        noise_map[2][seen] *= scale
        unseen = hit == 0
        noise_map[0][unseen] = hp.UNSEEN
        noise_map[1][unseen] = hp.UNSEEN
        noise_map[2][unseen] = hp.UNSEEN

    header += hitheader

    return sky_map, noise_map


def add_maps(args, sky_map, noise_map, header):
    
    print('Adding maps')

    if np.atleast_1d(sky_map).size == 1:
        print('  Sky map is empty, using the noise map')
        return noise_map

    if np.atleast_1d(noise_map).size == 1:
        print('  Noise map is empty, using the sky map')
        return sky_map

    total_map = sky_map.copy()
    good = total_map != hp.UNSEEN
    total_map[good] += noise_map[good]

    return total_map


def read_noise(args, header):
    """
    Parse the noise parameters into noise spectra and per pixel noise
    levels.  Zero knee multipole produces a white noise spectrum.
    """

    if (not args.noise_tt_sigma and not args.noise_ee_sigma
        and not args.noise_bb_sigma) or args.noise_scale == 0:
        return None, 0, 0

    print('Parsing noise')

    noiseheader = []

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
        knee = args.noise_tt_knee
    else:
        knee = 0
    slope = args.noise_tt_slope
    if knee:
        noise_cl[0][1:] = (1 + (ell[1:]/knee)**slope) * sigma**2
    else:
        noise_cl[0][1:] = sigma**2
    sigma_tt = sigma
    noiseheader.append(('ttsigma', sigma/norm, 'TT noise level [uK/arcmin]'))
    noiseheader.append(('ttknee', knee, 'TT noise knee'))
    noiseheader.append(('ttslope', slope, 'TT noise slope'))

    # EE noise spectrum (use TT parameters when EE are absent)

    if args.noise_ee_sigma:
        sigma = args.noise_ee_sigma * norm
    if args.noise_ee_knee is not None:
        knee = args.noise_ee_knee
    if args.noise_ee_slope:
        slope = args.noise_ee_slope
    if knee:
        noise_cl[1][2:] = (1 + (ell[2:]/knee)**slope) * sigma**2
    else:
        noise_cl[1][2:] = sigma**2
    sigma_ee = sigma
    noiseheader.append(('eesigma', sigma/norm, 'EE noise level [uK/arcmin]'))
    noiseheader.append(('eeknee', knee, 'EE noise knee'))
    noiseheader.append(('eeslope', slope, 'EE noise slope'))

    # BB noise spectrum (use EE parameters when BB are absent)

    if args.noise_bb_sigma:
        sigma = args.noise_bb_sigma * norm
    if args.noise_bb_knee is not None:
        knee = args.noise_bb_knee
    if args.noise_bb_slope:
        slope = args.noise_bb_slope
    if knee:
        noise_cl[2][2:] = (1 + (ell[2:]/knee)**slope) * sigma**2
    else:
        noise_cl[2][2:] = sigma**2
    sigma_bb = sigma
    noiseheader.append(('bbsigma', sigma/norm, 'BB noise level [uK/arcmin]'))
    noiseheader.append(('bbknee', knee, 'BB noise knee'))
    noiseheader.append(('bbslope', slope, 'BB noise slope'))

    if args.debug:
        fn = args.output + '_noise_cl.fits'
        if os.path.isfile(fn):
            os.remove(fn)
        print('  Writing N_ell to', fn)
        hp.write_cl(fn, list(noise_cl))

    # Split the noise into white and 1/f parts.  White noise is best
    # simulated in the pixel domain.

    nside = args.nside
    npix = hp.nside2npix(nside)
    wpix = np.sqrt(4*np.pi/npix)

    noise_cl[0][1:] -= sigma_tt**2
    sigma_tt /= wpix

    sigma_pol = min(sigma_ee, sigma_bb)
    noise_cl[1][2:] -= sigma_pol**2
    noise_cl[2][2:] -= sigma_pol**2
    sigma_pol /= wpix

    if not args.noise_tt_knee and not args.noise_ee_knee \
            and not args.noise_bb_knee:
        noise_cl = None

    return noise_cl, sigma_tt, sigma_pol, noiseheader


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
    print('  Reading', fn)
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
    print('  lmax =', lmax)

    alms = np.array(alms)

    if zero_dipole:
        print('  Removing TT monopole and dipole')
        for ell in range(2):
            for m in range(ell+1):
                i = hp.Alm.getidx(lmax, ell, m)
                alms[0][i] = 0

    return alms, lmax


def read_cmb(args, header):
    """
    Read (and co-add) the CMB alm expansion.
    """

    if args.cmb_scale == 0:
        # No CMB required
        return 0, 0

    print('Reading CMB')

    # Read CMB expansions

    cmbs = []
    lmaxs = []
    cmbheader = []

    for i, fn in enumerate(args.cmb):
        fn.format(args.realization)
        if not os.path.isfile(fn):
            raise RuntimeError('CMB file not found: {}'.format(fn))
        cmb, lmax = read_alm(fn, zero_dipole=True)
        cmbs.append(cmb)
        lmaxs.append(lmax)
        cmbheader.append(('cmb{}'.format(i), fn, 'CMB a_lm file'))
        cmbheader.append(('cmb{}lmax'.format(i), lmax, 'CMB a_lm lmax'))

    # Combine the expansions

    if len(cmbs) == 0:
        # No CMB loaded
        return 0, 0

    if args.lmax:
        lmaxtot = args.lmax
    else:
        lmaxtot = max(lmaxs)
    cmbheader.append(('cmblmax', lmaxtot, 'CMB a_lm lmax'))
    cmbheader.append(('cmbcoord', lmaxtot, 'CMB a_lm coord'))

    nalm = hp.Alm.getsize(lmaxtot)
    cmbtot = np.zeros([3, nalm], dtype=np.complex)

    for ell in range(lmaxtot+1):
        for m in range(ell+1):
            i0 = hp.Alm.getidx(lmaxtot, ell, m)
            for cmb, lmax in zip(cmbs, lmaxs):
                if ell <= lmax:
                    i1 = hp.Alm.getidx(lmax, ell, m)
                    cmbtot[:, i0] += cmb[:, i1]

    if args.cmb_coord != args.output_coord:
        print('  Rotating {} -> {}'.format(args.cmb_coord, args.output_coord))
        psi, theta, phi = coordsys2euler_zyz(args.cmb_coord, args.output_coord)
        hp.rotate_alm(cmbtot, psi, theta, phi)

    if args.cmb_scale and args.cmb_scale != 1:
        print('  Scaling CMB by', args.cmb_scale)
        cmb *= args.cmb_scale
        cmbheader.append(('cmbscale', args.cmb_scale, 'CMB scale'))

    header += cmbheader

    return cmbtot, lmaxtot


def read_foreground(args, header):
    """
    Read the foregrounds and expand in spherical harmonics.
    """

    if args.fg_scale == 0:
        # No foregrounds required
        return 0, 0

    print('Reading foregrounds')

    fgheader = []

    fg_tot = None
    for i, fn in enumerate(args.fg):
        if not os.path.isfile(fn):
            raise RuntimeError('Foreground file not found: {}'.format(fn))
        print('  Reading', fn)
        fg_map = np.array(hp.read_map(fn, range(3), verbose=False))
        if fg_tot is None:
            fg_tot = fg_map
        else:
            if fg_map.size != fg_tot.size:
                raise RuntimeError('Foreground maps have different resolution.')
            fg_tot += fg_map
        fgheader.append(('fg{:02}'.format(i), fn, 'FG map file'))

    if fg_tot is None:
        return 0, 0

    if args.fg_scale and args.fg_scale != 1:
        print('  Scaling foreground by', args.fg_scale)
        fg_tot[fg_tot != hp.UNSEEN] *= args.fg_scale
        fgheader.append(('fgscale', args.fg_scale, 'Foreground scale'))

    nside = hp.npix2nside(fg_tot[0].size)
    print('  nside =', nside)
    fgheader.append(('fgnside', nside, 'Foreground Nside'))
    fgheader.append(('fgcoord', args.fg_coord, 'Foreground coord'))

    if args.lmax:
        lmax = args.lmax
    else:
        lmax = 2*nside
    fgheader.append(('fglmax', lmax, 'Foreground a_lm lmax'))

    print('  Expanding foregrounds in spherical harmonics. lmax =', lmax)

    fg = hp.map2alm(fg_tot, lmax=lmax, pol=True, iter=0)
    fg = np.array(fg)

    # Invert and deconvolve the pixel window function

    print('  Deconvolving pixel window function.')

    pixwin = np.array(hp.pixwin(nside, pol=True))
    pixwin[pixwin != 0] = 1 / pixwin[pixwin != 0]

    hp.almxfl(fg[0], pixwin[0], inplace=True)
    hp.almxfl(fg[1], pixwin[1], inplace=True)
    hp.almxfl(fg[2], pixwin[1], inplace=True)

    if args.fg_coord != args.output_coord:
        print('  Rotating {} -> {}'.format(args.fg_coord, args.output_coord))
        psi, theta, phi = coordsys2euler_zyz(args.fg_coord, args.output_coord)
        hp.rotate_alm(fg, psi, theta, phi)

    header += fgheader

    return fg, lmax


def build_highpass(args, cmb_lmax, fg_lmax, header):
    """
    Build a harmonic highpass filter to mimic effects of TOD filtering.
    """

    if not args.lmin:
        return None

    print('Building the highpass filter. lmin =', args.lmin)

    hpheader = []

    if args.lmax:
        lmax = args.lmax
    else:
        lmax = max(cmb_lmax, fg_lmax)

    highpass = np.ones(lmax+1)
    highpass[:args.lmin] = 0

    hpheader.append(('lmin', args.lmin, 'highpass lmin'))

    if args.debug:
        fn = args.output + '_highpass_filter.fits'
        if os.path.isfile(fn):
            os.remove(fn)
        print('  Writing highpass filter to', fn)
        hp.write_cl(fn, highpass)

    header += hpheader

    return highpass


def smooth_signal(args, cmb_alm, cmb_lmax, fg_alm, fg_lmax, highpass, header):
    """
    Apply beam smoothing to the CMB and foreground expansions and return
    a co-added expansion.
    """

    if cmb_lmax == 0 and fg_lmax == 0:
        return 0

    if args.lmax:
        lmax = args.lmax
    else:
        lmax = max(cmb_lmax, fg_lmax)

    print('Combining and smoothing the alm expansions.')
    print('  fwhm =', args.fwhm, 'arcmin')
    print('  nside =', args.nside)

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

    if highpass is not None:
        print('  highpass filtering the signal')
        hp.almxfl(alm[0], highpass, inplace=True)
        hp.almxfl(alm[1], highpass, inplace=True)
        hp.almxfl(alm[2], highpass, inplace=True)

    signal = np.array(hp.alm2map(
            list(alm), args.nside, pixwin=True, verbose=False,
            fwhm=args.fwhm*arcmin, pol=True))

    header.append(('fwhm', args.fwhm, 'Beam width [arcmin]'))

    if args.debug:
        fn = args.output + '_signal_map.fits.gz'
        if os.path.isfile(fn):
            os.remove(fn)
        print('  Writing signal map to', fn)
        hp.write_map(fn, signal, coord=args.output_coord, extra_header=header)

    return signal


def save_map(args, total_map, header):
    """
    Save the co-added map as a zip-compressed full sky FITS file.
    """

    if np.atleast_1d(total_map).size == 1:
        print('\nWARNING: total map is empty. Nothing written.\n')
        return

    fn = args.output + '_map.fits.gz'

    if os.path.isfile(fn):
        os.remove(fn)

    print('Saving final map to', fn)

    hp.write_map(fn, total_map, coord=args.output_coord, extra_header=header)

    return


def simulate_noise(args, noise_cl, sigma_tt, sigma_pol, highpass,
                   header, noiseheader):
    """
    Simulate a noise map using the N_ell and white noise levels.
    We will assume diagonal white noise covariance matrices for now.
    """

    if args.noise_scale == 0:
        return 0

    print('Simulating noise')

    seed = args.seed
    while seed < 10000:
        seed += 1000
    seed = seed*(args.freq+1) + args.realization
    noiseheader.append(('seed1', args.seed, 'Input RNG seed'))
    noiseheader.append(('seed2', seed, 'Actual RNG seed'))
    np.random.seed(seed)

    npix = hp.nside2npix(args.nside)

    if noise_cl is not None:
        if args.lmax:
            lmax = args.lmax
        else:
            lmax = 2*args.nside

        print('  Simulating 1/f part in harmonic space. lmax =', lmax)

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

        if highpass is not None:
            print('  highpass filtering the 1/f noise')
            hp.almxfl(noise_alm[0], highpass, inplace=True)
            hp.almxfl(noise_alm[1], highpass, inplace=True)
            hp.almxfl(noise_alm[2], highpass, inplace=True)

        # Transform to pixel domain

        noise_map = np.array(hp.alm2map(
                list(noise_alm), args.nside, pixwin=False,
                verbose=False, pol=True, inplace=True))
    else:
        if not sigma_tt and not sigma_pol:
            return 0
        else:
            noise_map = np.zeros([3, npix])

    # Add white noise in pixel space

    print('  Simulating white part in pixel space.')

    noise_map[0] += np.random.randn(npix)*sigma_tt
    noise_map[1] += np.random.randn(npix)*sigma_pol
    noise_map[2] += np.random.randn(npix)*sigma_pol

    if args.noise_scale and args.noise_scale != 1:
        print('  Scaling noise by', args.noise_scale)
        noise_map *= args.noise_scale
        noiseheader.append(('noisescl', args.noise_scale, 'Noise scale'))

    if args.debug:
        fn = args.output + '_noise_map.fits.gz'
        if os.path.isfile(fn):
            os.remove(fn)
        print('  Writing noise map to', fn)
        hp.write_map(fn, noise_map, coord=args.output_coord,
                     extra_header=noiseheader)

    header += noiseheader

    return noise_map


def main():

    header = []

    args = parse_arguments(header)

    cmb_alm, cmb_lmax = read_cmb(args, header)

    fg_alm, fg_lmax = read_foreground(args, header)

    highpass = build_highpass(args, cmb_lmax, fg_lmax, header)

    sky_map = smooth_signal(args, cmb_alm, cmb_lmax, fg_alm, fg_lmax,
                            highpass, header)

    noise_cl, sigma_tt, sigma_pol, noiseheader = read_noise(args, header)

    noise_map = simulate_noise(args, noise_cl, sigma_tt, sigma_pol,
                               highpass, header, noiseheader)

    sky_map, noise_map = apply_hits(args, sky_map, noise_map, header)

    total_map = add_maps(args, sky_map, noise_map, header)

    save_map(args, total_map, header)

    return


if __name__ == '__main__':

    main()
