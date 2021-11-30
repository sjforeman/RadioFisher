#!/usr/bin/python
"""
Generate a baseline distribution and output binned baseline distances.

Uses parameters for CHIME.
"""
import os
import numpy as np
import pylab as P
import scipy.spatial

outdir = "array_config"
outfile = os.path.join(outdir, "CHIME_baselines")

# Number of cylinders
Ncyl = 4
# Distance between adjacent cylinders, measured center-to-center in m
w_cyl = 22
# Cylinder length in m (not actually used for baseline distribution)
l_cyl = 100
# Number of feeds per cylinder
Nfeeds = 256 
# Separation between feeds, in m
d_feed = 0.3048

# Layout receivers
x = []; y = []
for i in range(Ncyl):
    for j in range(Nfeeds):
        xx = i * w_cyl
        yy = j * d_feed
        #P.plot(xx, yy, 'bx')
        x.append(xx); y.append(yy)

# Calculate baseline separations
d = scipy.spatial.distance.pdist( np.column_stack((x, y)) )

# Create output directory if it doesn't exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Save baselines
np.save(outfile, d)

# Output stats
print("Cylinders:                         %d" % Ncyl)
print("Feeds/cyl.:                        %d" % Nfeeds)
print("Tot. feeds:                        %d" % (Ncyl * Nfeeds))
print("Center-to-center cyl. separation:  %g m" % w_cyl)
print("Cycl. length:                      %g m" % l_cyl)
print("Feed sep.:                         %g m" % d_feed)
print("Max baseline:                      %g m" % np.max(d))
print("-"*50)
print("Output file:                       %s.npy" % outfile)

#P.hist(d, bins=200)
#P.show()

