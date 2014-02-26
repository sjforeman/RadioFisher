import numpy as np
import scipy.interpolate
from units import *

# Define fiducial cosmology and parameters
# Planck-only best-fit parameters, from Table 2 of Planck 2013 XVI.
cosmo = {
    'omega_M_0':        0.316,
    'omega_lambda_0':   0.684,
    'omega_b_0':        0.049,
    'omega_HI_0':       6.5e-4, # 9.4e-4
    'N_nu':             0,
    'h':                0.67,
    'ns':               0.962,
    'sigma_8':          0.834,
    'gamma':            0.55,
    'w0':               -1.,
    'wa':               0.,
    'fNL':              0.,
    'mnu':              0.,
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             0.702,
    'A':                1.,
    'sigma_nl':         7.,
    'beta_1':           0.,         # Scale-dependent bias (k^1 term coeff. [Mpc])
    'beta_2':           0.          # Scale-dependent bias (k^2 term coeff. [Mpc^2])
}

# Define which measurements to include in forecasts
USE = {
  'f_rsd':             True,     # RSD constraint on f(z)
  'f_growthfactor':    False,    # D(z) constraint on f(z)
  'alpha_all':         False,     # Use all constraints on alpha_{perp,par}
  'alpha_volume':      False,
  'alpha_rsd_angle':   False, #t
  'alpha_rsd_shift':   False, #t
  'alpha_bao_shift':   True,
  'alpha_pk_shift':    False # True
}

SURVEY = {
    'ttot':             10e3*HRS_MHZ,      # Total integration time [MHz^-1]
    'Sarea':            30e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'use':              USE                # Which constraints to use/ignore
}

# Add foreground components to cosmology dict.
# (Extragal. ptsrc, extragal. free-free, gal. synch., gal. free-free)
foregrounds = {
    'A':     [57.0, 0.014, 700., 0.088],        # FG noise amplitude [mK^2]
    'nx':    [1.1, 1.0, 2.4, 3.0],              # Angular scale powerlaw index
    'mx':    [-2.07, -2.10, -2.80, -2.15],      # Frequency powerlaw index
    'l_p':   1000.,                             # Reference angular scale
    'nu_p':  130.                               # Reference frequency [MHz]
}
cosmo['foregrounds'] = foregrounds


################################################################################
# Illustrative experiments used in paper
################################################################################

exptS = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            50,                # No. of beams (for multi-pixel detectors)
    'Ddish':            30.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    }
exptS.update(SURVEY)

exptM = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            160, #128              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            4.,                # Single dish diameter [m]
    'Tinst':            35.*(1e3),   #35      # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000., #800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Dmax':             60.,              # Max. interferom. baseline [m]
    'Dmin':             4.                 # Min. interferom. baseline [m]
    }
exptM.update(SURVEY)

exptL = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,             # Bandwidth of single channel [MHz]
    'Dmax':             600.,              # Max. interferom. baseline [m]
    'Dmin':             15.                # Min. interferom. baseline [m]
    }
exptL.update(SURVEY)

#################################
# OLD VERSIONS
"""
exptS = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            50,                # No. of beams (for multi-pixel detectors)
    'Ddish':            30.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    }
exptS.update(SURVEY)

exptM = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            10.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Dmax':             100.,              # Max. interferom. baseline [m]
    'Dmin':             20.                # Min. interferom. baseline [m]
    }
exptM.update(SURVEY)

exptL = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':             nx['SKAMREF'],     # Interferometer antenna density
    'Dmax':             100.,              # Max. interferom. baseline [m]
    'Dmin':             20.                # Min. interferom. baseline [m]
    }
exptL.update(SURVEY)
"""

################################################################################
# Configurations from Mario's notes
################################################################################

GBT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            100.,              # Single dish diameter [m]
    'Tinst':            29.*(1e3),         # System temp. [mK]
    'survey_dnutot':    240.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     920.,              # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
GBT.update(SURVEY)

BINGO = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            50,                # No. of beams (for multi-pixel detectors)
    'Ddish':            30.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1260.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
BINGO.update(SURVEY)

# FIXME: What is the actual bandwidth of WSRT?
WSRT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            14,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            120.*(1e3),        # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1200.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
WSRT.update(SURVEY)

APERTIF = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            14,                # No. of dishes
    'Nbeam':            37,                # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            52.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1300.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
APERTIF.update(SURVEY)

JVLA = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            27,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            70.*(1e3),         # System temp. [mK]
    'survey_dnutot':    420.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)': "array_config/nx_VLAD_dec90.dat" # Interferometer antenna density
    }
JVLA.update(SURVEY)

ASKAP = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            36,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':            12.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)': "array_config/nx_ASKAP_dec30.dat" # Interferometer antenna density
    }
ASKAP.update(SURVEY)

KAT7 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            7,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    220.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)': "array_config/nx_KAT7_dec30.dat" # Interferometer antenna density
    }
KAT7.update(SURVEY)

# NB: For MeerKAT Band 1 only.
MeerKAT_band1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            64,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    520.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)': "array_config/nx_MKREF2_dec30.dat" # Interferometer antenna density
    }
MeerKAT_band1.update(SURVEY)

# NB: For MeerKAT Band 2 only.
MeerKAT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            64,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)': "array_config/nx_MKREF2_dec30.dat" # Interferometer antenna density
    }
MeerKAT.update(SURVEY)

SKA1MID = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)': "array_config/nx_SKAM190_dec30.dat" # Interferometer antenna density
    }
SKA1MID.update(SURVEY)

superSKA1MID = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            254, #190,         # No. of dishes
    'Nbeam':            1,  # NOTE!        # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)': "array_config/nx_SKAM190_dec30.dat" # Interferometer antenna density
    }
superSKA1MID.update(SURVEY)

# For SKA1-SUR band 2 only
SKA1SUR = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            60,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150.,             # Max. freq. of survey
    'dnu':              0.005              # Bandwidth of single channel [MHz]
    }
SKA1SUR.update(SURVEY)

SKA1SUR_band1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            60,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     850.,              # Max. freq. of survey
    'dnu':              0.005              # Bandwidth of single channel [MHz]
    }
SKA1SUR_band1.update(SURVEY)

# Surveys that are fefined as overlap between two instruments
SKAMID_PLUS = {
    'overlap':          [SKA1MID, MeerKAT],
    'n(x)':             "array_config/nx_SKAMREF2COMP_dec30.dat"
    }

SKAMID_PLUS_band1 = {
    'overlap':          [SKA1MID, MeerKAT_band1],
    'n(x)':             "array_config/nx_SKAMREF2COMP_dec30.dat"
    }

SKASUR_PLUS = {
    'overlap':          [SKA1SUR, ASKAP]
    }

SKASUR_PLUS_band1 = {
    'overlap':          [SKA1SUR_band1, ASKAP]
    }
