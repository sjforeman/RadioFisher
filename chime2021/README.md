CHIME vs. DESI BAO Forecasts
----------------------------

This directory contains files related to the BAO forecasts for CHIME and DESI
shown in the CHIME instrument overview paper. I confine the ingredients to this
directory as much as possible, but unfortunately, some changes to the main
RadioFisher code were necessary. These changes should be obvious if you look
at the commit history for this branch.

To run the forecasts:

 - Generate the CHIME baseline distribution with
   `generate_chime_baseline_distribution.py`.
 - Bin the baseline distribution with `process_chime_baselines.py`.
 - Generate the DESI Fisher matrix with `galaxy_full_experiment.py`.
 - Generate the CHIME Fisher matrix with `full_experiment.py`.
 - Generate the D_V(z) errorbar forecast plot with `plot_dv_forecasts.py`.

These forecasts are the product of combined efforts by Tianyue Chen and Simon
Foreman.
