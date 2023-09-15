#!/usr/bin/env python

import freegs
import numpy as np
import matplotlib.pyplot as plt

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.TestTokamak()

eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-1.0, Zmax=1.0,   # Height range
                        nx=65, ny=65,          # Number of grid points
                        boundary=freegs.boundary.freeBoundaryHagenow)  # Boundary condition


#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainBetapIp(eq,
                                        3.214806e-02, # Poloidal beta
                                        2e5, # Plasma current [Amps]
                                        2.0) # Vacuum f=R*Bt

#########################################
# Coil current constraints
#
# Specify locations of the X-points
# to use to constrain coil currents

# Define desired plasma shape
R0 = 1.27
Z0 = 0.00
A = 3.1
tri_u = 0.45
tri_l = 0.45
elon_u = 1.6
elon_l = 1.6
fsep_u = 1 # 1 for xpoint, 0 for no xpoint
fsep_l = 1

# Generate LCFS dictionary for the specified plasma shape
LCFS = freegs.control.generate_separatrix_points(R0,A,Z0,tri_u,tri_l,elon_u,elon_l,fsep_u,fsep_l)

# Set xpoints - Note: still need to include the additional xpoint above the plasma that is not on the LCFS
xpoints = [LCFS['LOWER'],   # (R,Z) locations of X-points -> DN
           LCFS['UPPER']]

# Set isoflux constraints - (R1,Z1, R2,Z2) pair of locations
isoflux = [(LCFS['IMP'][0],LCFS['IMP'][1], LCFS['OMP'][0],LCFS['OMP'][1]), # IMP and OMP
           (LCFS['OMP'][0],LCFS['OMP'][1], LCFS['UPPER'][0],LCFS['UPPER'][1]), # OMP and UPPER point
           (LCFS['OMP'][0],LCFS['OMP'][1], LCFS['LOWER'][0],LCFS['LOWER'][1]) # OMP and LOWER point
           ]

constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

#########################################
# Nonlinear solve

freegs.solve(eq,          # The equilibrium to adjust
             profiles,    # The toroidal current profile function
             constrain,
             show=True)   # Constraint function to set coil currents

# eq now contains the solution

print("Done!")

print("Plasma current: %e Amps" % (eq.plasmaCurrent()))
print("Plasma pressure on axis: %e Pascals" % (eq.pressure(0.0)))
print("Poloidal beta: %e" % (eq.poloidalBeta()))

# Currents in the coils
tokamak.printCurrents()

# Forces on the coils
eq.printForces()

print("\nSafety factor:\n\tpsi \t q")
for psi in [0.01, 0.9, 0.95]:
    print("\t{:.2f}\t{:.2f}".format(psi, eq.q(psi)))

##############################################
# Final plot of equilibrium

axis = eq.plot(show=False)
eq.tokamak.plot(axis=axis, show=False)
constrain.plot(axis=axis, show=True)

# Safety factor

import matplotlib.pyplot as plt
plt.plot(*eq.q())
plt.xlabel(r"Normalised $\psi$")
plt.ylabel("Safety factor")
plt.show()