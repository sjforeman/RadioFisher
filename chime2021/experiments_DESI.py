"""
Experimental settings for DESI.

Modified version of radiofisher/experiments_galaxy.py
"""
import numpy as np
from scipy.interpolate import interp1d

from rfwrapper import rf
import radiofisher
import experiments_CHIME as experiments

def sarea_to_fsky(sarea):
    """
    Convert a survey area, in deg^2, into a sky fraction, f_sky.
    """
    FULLSKY = 4.*np.pi * (180./np.pi)**2.
    return sarea / FULLSKY

def load_expt(expt):
    """
    Process experiment dict to load fields from file.
    """
    # No action taken if 'fname' not specified (warn if loadable fields exist)
    if 'fname' not in list(expt.keys()):
        flagged_fields = False
        for key in list(expt.keys()):
            if key[0] == '_': flagged_fields = True
        if flagged_fields:
            print("\tload_expt(): No filename specified; couldn't load some fields.")
    else:
        # Load fields that need to be loaded
        dat = np.genfromtxt(expt['fname']).T
        for key in list(expt.keys()):
            if key[0] == '_':
                expt[key[1:]] = dat[expt[key]]

    # Process bias
    if 'nz' not in list(expt.keys()):
        zc = 0.5 * (expt['zmin'] + expt['zmax'])
        expt['nz'] = expt['n(z)'](zc)

    # Rescale n(z) if requested
    if 'rescale_nz' in list(expt.keys()): expt['nz'] *= expt['rescale_nz']

    # Process bias
    if 'b' not in list(expt.keys()):
        zc = 0.5 * (expt['zmin'] + expt['zmax'])
        expt['b'] = expt['b(z)'](zc)
    return expt


# Define which measurements to include in forecasts.
# Turn most of them off for our BAO-specific forecasts.
USE = {
  'f_rsd':             False,     # RSD constraint on f(z)
  'f_growthfactor':    False,    # D(z) constraint on f(z)
  'alpha_all':         False,     # Use all constraints on alpha_{perp,par}
  'alpha_volume':      False,
  'alpha_rsd_angle':   False,
  'alpha_rsd_shift':   False,
  'alpha_bao_shift':   True,
  'alpha_pk_shift':    False
}

# Generic survey parameters
SURVEY = {
    'kmin':     1e-4,   # Seo & Eisenstein say: shouldn't make much difference...
    'k_nl0':    0.14,   # Non-linear scale at z=0 (effective kmax)
    'use':      USE
}


#########################
# DESI_combined
#########################

cosmo = experiments.cosmo
cosmo_fns = radiofisher.background_evolution_splines(cosmo)
# H(z), Hubble rate in km/s/Mpc
# r(z), comoving distance in Mpc
# D(z), linear growth factor
# f(z), linear growth rate
H, r, D, f = cosmo_fns

def comoving_volume(zmin, zmax):
    # Comoving volume between zmin and zmax
    return 4 * np.pi / 3 * (r(zmax)**3 - r(zmin)**3)

# Compute redshift bins matching those used in Aghamousa et al. 2016
zmin_desi = np.concatenate([np.arange(0, 0.41, 0.1), np.arange(0.6, 1.81, 0.1)])
zmax_desi = zmin_desi + 0.1
zcenter_desi = 0.5 * (zmin_desi + zmax_desi)

# Load DESI redshift distributions, in units of gal / dz / deg^2
desi_raw_dist = {}
desi_raw_dist['bgs'] = np.loadtxt('forecast_inputs/desi_bgs_nz_table2.3.dat').T
desi_other_dist = np.loadtxt('forecast_inputs/desi_lrg_elg_qso_nz_table2.3.dat').T
for s, c in zip(['elg', 'lrg', 'qso'], [1, 2, 3]):
    desi_raw_dist[s] = desi_other_dist[[0, c]]

# To convert the raw DESI redshift distributions to Mpc^-3,
# we first multiply by dz=0.1 and Asky=14000deg^2, then divide by the comoving
# volume in each z bin.
desi_nz_sample_mpc3 = {}
for s in ['bgs', 'lrg', 'elg', 'qso']:
    desi_nz_sample_mpc3[s] = np.zeros_like(desi_raw_dist[s])
    desi_nz_sample_mpc3[s][0] = desi_raw_dist[s][0]

    for i in range(desi_nz_sample_mpc3[s].shape[1]):
        z = desi_nz_sample_mpc3[s][0, i]
        n_raw = desi_raw_dist[s][1, i]

        dz = 0.1
        Asky = 14000
        fsky = Asky / (360**2/np.pi)
        vol = comoving_volume(z-0.05,z+0.05) * fsky

        desi_nz_sample_mpc3[s][1,i] = n_raw * dz * Asky / vol

# Compute individual n(z)'s over full z range we care about,
# and also compute their sum
nz_desi_sample_fullz = {}
nz_desi = np.zeros_like(zcenter_desi)
for s in ['bgs', 'lrg', 'elg', 'qso']:
    temp_nz_interp = interp1d(
        desi_nz_sample_mpc3[s][0], desi_nz_sample_mpc3[s][1], bounds_error=False, fill_value=0
    )
    nz_desi_sample_fullz[s] = temp_nz_interp(zcenter_desi)
    nz_desi += nz_desi_sample_fullz[s]

# Compute b_g(z) for each sample, based on forms in Sec. 2.4.2
# of Aghamousa et al. 2016
bz_desi_sample_normalization = {
    "bgs": 1.34, "lrg": 1.7, "elg": 0.84, "qso": 1.2
}
bz_desi_sample = {}
for s, norm in bz_desi_sample_normalization.items():
    bz_desi_sample[s] = norm / D(zcenter_desi)

# Compute n(z)-weighted average of b_g(z)
bz_desi = np.array(
    [bz_desi_sample[s] * nz_desi_sample_fullz[s] for s in ['bgs', 'lrg', 'elg', 'qso']]
).sum(axis=0) / np.array(
    [nz_desi_sample_fullz[s]  for s in ['bgs', 'lrg', 'elg', 'qso']]
).sum(axis=0)
bz_desi = np.nan_to_num(bz_desi)

# print(zcenter_desi, bz_desi)

# Set DESI survey params, using 14000 deg^2 for baseline survey.
DESI_combined = {
    'fsky':        sarea_to_fsky(14e3),
    'zmin':        zmin_desi,
    'zmax':        zmax_desi,
    'nz':          nz_desi,
    'b':          bz_desi,
}
DESI_combined.update(SURVEY)
