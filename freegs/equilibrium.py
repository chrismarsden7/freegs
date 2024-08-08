"""
Defines class to represent the equilibrium
state, including plasma and coil currents
"""

from numpy import pi, meshgrid, linspace, exp, array
import numpy as np
from scipy import interpolate
from scipy.integrate import romb, cumulative_trapezoid  # Romberg integration
from scipy.optimize import leastsq
from scipy.interpolate import interp1d

from .boundary import fixedBoundary, freeBoundary
from . import critical

from . import polygons

# Operators which define the G-S equation
from .gradshafranov import mu0, GSsparse, GSsparse4thOrder

# Multigrid solver
from . import multigrid

from . import machine

import matplotlib.pyplot as plt

from shapely import intersection, LineString
from shapely.geometry import Polygon

class Equilibrium:
    """
    Represents the equilibrium state, including
    plasma and coil currents

    Data members
    ------------

    These can be read, but should not be modified directly

    R[nx,ny]
    Z[nx,ny]

    Rmin, Rmax
    Zmin, Zmax

    tokamak - The coils and circuits

    Private data members

    _applyBoundary()
    _solver - Grad-Shafranov elliptic solver
    _profiles     An object which calculates the toroidal current
    _constraints  Control system which adjusts coil currents to meet constraints
                  e.g. X-point location and flux values
    """

    def __init__(
        self,
        tokamak=machine.EmptyTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=freeBoundary,
        psi=None,
        current=0.0,
        order=4,
        check_limited=False,
    ):
        """Initialises a plasma equilibrium

        Rmin, Rmax  - Range of major radius R [m]
        Zmin, Zmax  - Range of height Z [m]

        nx - Resolution in R. This must be 2^n + 1
        ny - Resolution in Z. This must be 2^m + 1

        boundary - The boundary condition, either freeBoundary or fixedBoundary

        psi - Magnetic flux. If None, use concentric circular flux
              surfaces as starting guess

        current - Plasma current (default = 0.0)

        order - The order of the differential operators to use.
                Valid values are 2 or 4.

        check_limited - Boolean, checks if the plasma is limited.
        """

        self.tokamak = tokamak

        self._applyBoundary = boundary

        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nx = nx
        self.ny = ny

        self.R_1D = linspace(Rmin, Rmax, nx)
        self.Z_1D = linspace(Zmin, Zmax, ny)
        self.R, self.Z = meshgrid(self.R_1D, self.Z_1D, indexing="ij")

        self.dR = self.R[1, 0] - self.R[0, 0]
        self.dZ = self.Z[0, 1] - self.Z[0, 0]

        self.check_limited = check_limited
        self.is_limited = False
        self.Rlim = None
        self.Zlim = None

        self.psi_bndry = None

        if psi is None:
            # Starting guess for psi
            xx, yy = meshgrid(linspace(0, 1, nx), linspace(0, 1, ny), indexing="ij")
            psi = exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4 ** 2)

            psi[0, :] = 0.0
            psi[:, 0] = 0.0
            psi[-1, :] = 0.0
            psi[:, -1] = 0.0

        # Calculate coil Greens functions. This is an optimisation,
        # used in self.psi() to speed up calculations
        self._pgreen = tokamak.createPsiGreens(self.R, self.Z)

        self._current = current  # Plasma current
        self.Jtor = None

        self._updatePlasmaPsi(psi)  # Needs to be after _pgreen

        # Create the solver
        if order == 2:
            generator = GSsparse(Rmin, Rmax, Zmin, Zmax)
        elif order == 4:
            generator = GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)
        else:
            raise ValueError(
                "Invalid choice of order ({}). Valid values are 2 or 4.".format(order)
            )
        self.order = order

        self._solver = multigrid.createVcycle(
            nx, ny, generator, nlevels=1, ncycle=1, niter=2, direct=True
        )

    def setSolverVcycle(self, nlevels=1, ncycle=1, niter=1, direct=True):
        """
        Creates a new linear solver, based on the multigrid code

        nlevels  - Number of resolution levels, including original
        ncycle   - The number of V cycles
        niter    - Number of linear solver (Jacobi) iterations per level
        direct   - Use a direct solver at the coarsest level?

        """
        generator = GSsparse(self.Rmin, self.Rmax, self.Zmin, self.Zmax)
        nx, ny = self.R.shape

        self._solver = multigrid.createVcycle(
            nx,
            ny,
            generator,
            nlevels=nlevels,
            ncycle=ncycle,
            niter=niter,
            direct=direct,
        )

    def setSolver(self, solver):
        """
        Sets the linear solver to use. The given object/function must have a __call__ method
        which takes two inputs

        solver(x, b)

        where x is the initial guess. This should solve Ax = b, returning the result.

        """
        self._solver = solver

    def callSolver(self, psi, rhs):
        """
        Calls the psi solver, passing the initial guess and RHS arrays

        psi   Initial guess for the solution (used if iterative)
        rhs

        Returns
        -------

        Solution psi

        """
        return self._solver(psi, rhs)

    def getMachine(self):
        """
        Returns the handle of the machine, including coils
        """
        return self.tokamak

    def plasmaCurrent(self):
        """
        Plasma current [Amps]
        """
        return self._current

    def plasmaVolume(self,psiN=None):
        """
        Calculate the volume of the plasma in m^3 that is 
        enclosed within the flux surface given by psiN.

        psiN should be between 0 and 1.
        """

        # Volume elements
        dV_not_masked = 2.0 * pi * self.R * self.dR * self.dZ

        # Get psi on the grid
        psi = self.psi()

        # Get the opoint and primary xpoint psi
        opt, xpt = critical.find_critical(self.R, self.Z, psi)

        # If psiN is not specified, use the primary separatrix
        if psiN is None:

            psiN = 1.0

        if np.ndim(psiN) == 0:

            if psiN == 0.0:

                return 0.0

            else:

                # Get the psi of the specified psiN
                psi_target = self.psi_psiN(psiN)

                # Mask the core up to the flux surface of the specified psiN
                mask = critical.core_mask(
                    self.R, self.Z, psi, opt, xpt, psi_target
                )

                # Only include points in the mask
                dV = dV_not_masked * mask

                # Integrate volume in 2D
                return romb(romb(dV))

        else:

            vals = []

            for pn in psiN:

                if pn == 0.0:

                    vals.append(0.0)

                else:

                    # Get the psi of the specified psiN
                    psi_target = self.psi_psiN(pn)

                    # Mask the core up to the flux surface of the specified psiN
                    mask = critical.core_mask(
                        self.R, self.Z, psi, opt, xpt, psi_target
                    )

                    # Only include points in the mask
                    dV = dV_not_masked * mask

                    # Integrate volume in 2D
                    val = romb(romb(dV))

                    vals.append(val)

            return np.asarray(vals)

    def dV_dpsiN(self,psiN=None):
        """Calculate dV/dpsiN"""

        eps = 0.01

        if psiN is None:

            psiN = 1.0

        if np.ndim(psiN) == 0:

            if psiN <= eps:

                return self.plasmaVolume(eps)/eps

            elif psiN >= 1.0-eps:

                return (self.plasmaVolume(1.0)-self.plasmaVolume(1.0-eps))/eps
            
            else: # eps < psiN < 1 - eps

                return (self.plasmaVolume(psiN+eps)-self.plasmaVolume(psiN-eps))/(2.*eps)

        else:

            vals = []

            for pn in psiN:

                if pn <= eps:

                    val = self.plasmaVolume(eps)/eps

                elif pn >= 1.0-eps:

                    val = (self.plasmaVolume(1.0)-self.plasmaVolume(1.0-eps))/eps
                
                else: # eps < psiN < 1 - eps

                    val = (self.plasmaVolume(pn+eps)-self.plasmaVolume(pn-eps))/(2.*eps)

                vals.append(val)

            return np.asarray(vals)

    def plasmaBr(self, R, Z):
        """
        Radial magnetic field due to plasma
        Br = -1/R dpsi/dZ
        """
        return -self.psi_func(R, Z, dy=1, grid=False) / R

    def plasmaBz(self, R, Z):
        """
        Vertical magnetic field due to plasma
        Bz = (1/R) dpsi/dR
        """
        return self.psi_func(R, Z, dx=1, grid=False) / R

    def Br(self, R, Z):
        """
        Total radial magnetic field
        """
        return self.plasmaBr(R, Z) + self.tokamak.Br(R, Z)

    def Bz(self, R, Z):
        """
        Total vertical magnetic field
        """
        return self.plasmaBz(R, Z) + self.tokamak.Bz(R, Z)

    def Bpol(self, R, Z):
        """
        Total poloidal magnetic field
        """
        Br = self.Br(R, Z)
        Bz = self.Bz(R, Z)
        return np.sqrt(Br * Br + Bz * Bz)

    def Btor(self, R, Z):
        """
        Toroidal magnetic field
        """
        # Normalised psi
        psi_norm = self.psiNRZ(R,Z)

        # Get f = R * Btor in the core. May be invalid outside the core
        fpol = self.fpol(psi_norm)

        if self.mask is not None:
            # Get the values of the core mask at the requested R,Z locations
            # This is 1 in the core, 0 outside
            mask = self.mask_func(R, Z, grid=False)
            fpol = fpol * mask + (1.0 - mask) * self.fvac()

        return fpol / R

    def Btot(self, R, Z):
        """
        Total magnetic field
        """
        Br = self.Br(R, Z)
        Bz = self.Bz(R, Z)
        Btor = self.Btor(R, Z)
        return np.sqrt(Br * Br + Bz * Bz + Btor * Btor)

    def psi(self):
        """
        Total poloidal flux (psi), including contribution from
        plasma and external coils.
        """
        # return self.plasma_psi + self.tokamak.psi(self.R, self.Z)
        return self.plasma_psi + self.tokamak.calcPsiFromGreens(self._pgreen)

    def psiN(self):
        """
        Total poloidal flux (psi), including contribution from
        plasma and external coils. Normalised such that psiN = 0 on
        the magnetic axis and 1 on the LCFS.
        """
        # return self.plasma_psi + self.tokamak.psi(self.R, self.Z)
        return (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

    def psi_psiN(self,psiN):
        """
        Poloidal flux for a given value of normalised poloidal flux.
        """

        return psiN * (self.psi_bndry - self.psi_axis) + self.psi_axis

    def psiN_psi(self,psi):
        """
        Normalised poloidal flux for a given value of poloidal flux.
        """

        return (psi - self.psi_axis) / (self.psi_bndry - self.psi_axis)        

    def psiRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location
        """
        return self.psi_func(R, Z, grid=False) + self.tokamak.psi(R, Z)

    def psiNRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location. Normalised such
        that psiN = 0 on the magnetic axis and 1 on the LCFS.
        """
        return (self.psiRZ(R, Z) - self.psi_axis) / (self.psi_bndry - self.psi_axis)

    def rhoPol_psi(self, psi):
        """
        Return rho poloidal = sqrt(psiN) for a given psi
        """

        psiN = self.psiN_psi(psi)
        rho_pol = np.sqrt(psiN)

        return rho_pol

    def rhoPolRZ(self, R, Z):
        """
        Return rho poloidal = sqrt(psiN) for a point at a given R,Z
        """

        psiN = self.psiNRZ(R,Z)
        rho_pol = np.sqrt(psiN)

        return rho_pol

    def fpol(self, psinorm):
        """
        Return f = R*Bt at specified values of normalised psi
        """
        return self._profiles.fpol(psinorm)

    def fvac(self):
        """
        Return vacuum f = R*Bt
        """
        return self._profiles.fvac()

    def q_axis(self):
        '''
        Calculates the safety factor on the magnetic axis. This uses the definition
        in Friedberg's Ideal MHD book

        q0 = (Btor_ax / (mu R_ax Jtor_ax) ) * ( (1 + kappa_0**2) / kappa_0)

        Where the subscript ax are quantities calculated at the magnetic axis
        and kappa_0 is the on axis elongation, defined as

        kappa_0**2 = (d2 psi / dR2) / (d2 psi / dZ2) where d2 psi / dx2 is the
        second derivative of the poloidal magnetic flux psi, at the magnetic axis
        '''

        # Coordinate of the magnetic axis
        Rax = self.Rmagnetic()
        Zax = self.Zmagnetic()

        # Toroidal field
        Btor_ax = self.Btor(Rax,Zax)

        # Current density
        Jtor_ax = self.JtorRZ(Rax,Zax)

        # Intermediate variable
        alpha = Btor_ax / (mu0*Rax*Jtor_ax)

        # Second derivatives of psi wrt R and Z
        d2_psi_dR2_ax = self.psi_func(Rax,Zax,dx=2,grid=False)
        d2_psi_dZ2_ax = self.psi_func(Rax,Zax,dy=2,grid=False)

        # On axis elongation
        kappa_ax = np.sqrt(d2_psi_dR2_ax/d2_psi_dZ2_ax)

        # On axis safety factor
        q0 = alpha*((1.+kappa_ax*kappa_ax)/kappa_ax)

        return q0

    def q(self, psiN=None, npoints=1000):
        '''
        Calculates the safety factor q at specified values of normalised psi.

        q = (1/ 2pi) * int( (Btor/(R*Bpol) * dl) )
        '''

        if psiN is None:

            psiN = 0.95

        if np.ndim(psiN) == 0:

            if psiN == 0:
                q = self.q_axis()

            else:

                Rsurf, Zsurf = self.psi_surfRZ(psiN, npoints=npoints)

                # Get the poloidal field along the surface
                Bpol_surf = self.Bpol(Rsurf,Zsurf)

                # Get the toroidal field along the surface
                Btor_surf = self.Btor(Rsurf,Zsurf)

                # Get dl along the surface
                dl = np.sqrt(np.diff(Rsurf)**2.0 + np.diff(Zsurf)**2.0)
                dl = np.insert(dl,0,0.0)

                # Get l along the surface
                l = np.cumsum(dl)

                # Integrand
                I = Btor_surf / (Rsurf*Bpol_surf)

                q = np.trapz(y=I,x=l) / (2.0 * np.pi)

            return q

        else:

            vals = []

            for pn in psiN:

                if pn == 0:
                    q = self.q_axis()

                else:

                    Rsurf, Zsurf = self.psi_surfRZ(pn, npoints=npoints)

                    # Get the poloidal field along the surface
                    Bpol_surf = self.Bpol(Rsurf,Zsurf)

                    # Get the toroidal field along the surface
                    Btor_surf = self.Btor(Rsurf,Zsurf)

                    # Get dl along the surface
                    dl = np.sqrt(np.diff(Rsurf)**2.0 + np.diff(Zsurf)**2.0)
                    dl = np.insert(dl,0,0.0)

                    # Get l along the surface
                    l = np.cumsum(dl)

                    # Integrand
                    I = Btor_surf / (Rsurf*Bpol_surf)

                    q = np.trapz(y=I,x=l) / (2.0 * np.pi)

                vals.append(q)

            return vals

    def phi_psi(self, psi):
        """
        Calculates toroidal flux, phi, at specified values of poloidal flux, psi.
        q = dphi/dpsi
        \phi(\psi=\psi_{i}) = \int_{\psi_{i}}^{\psi_{ax}} q \,d\psi
        \phi(\psi=\psi_{i}) = - \int_{\psi_{ax}}^{\psi_{i}} q \,d\psi

        """

        # As the user may elect to calculate phi at a single value of psi, we must first
        # prepare many points in psi, and use these to build a profile for phi=phi(psi),
        # which can later be used to interpolate out phi at the specified psi.

        # psiN of these points
        psiN_vals = np.linspace(0.0,1.0,101)

        # Corresponding psi of these points
        psi_vals = self.psi_psiN(psiN_vals)

        # Safety factor of these points
        q_vals = self.q(psiN_vals)

        # Integrate q wrt psi to get the toroidal flux, phi. phi = 0 @ psiN = 0
        phi_vals = -1.0*cumulative_trapezoid(y=q_vals,x=psi_vals, initial=0.0)

        # Create 1D interpolator for phi=phi(psi)
        phi_func = interp1d(psi_vals,phi_vals,kind='linear',fill_value='extrapolate',bounds_error=False)

        # Obtain phi at the user-specified psi
        result = phi_func(psi)

        return result

    def phiRZ(self,R,Z):
        """
        Calculate toroidal flux, phi, at specified values of R,Z.
        """

        # Get the psi at the specified R,Z
        psi = self.psiRZ(R,Z)

        # Get the toroidal flux, phi.
        phi = self.phi_psi(psi)

        return phi

    def phiN_psi(self, psi=None):
        """
        Calculates the normalised toroidal flux, phiN, at specified values of psi.

        phiN = (phi - phi_ax) / (phi_bndry - phi_ax)
        phiN = phi/phi_bndry

        as phi_ax = 0 by definition
        """

        # Get phi_bndry == phi(psi_bndry)
        phi_bndry = self.phi_psi(self.psi_bndry)

        # Get phi at the user-specified psi
        phi = self.phi_psi(psi)

        phiN = phi/phi_bndry

        return phiN

    def phiNRZ(self, R,Z):
        """
        Calculates the normalised toroidal flux, phiN, at specified values of R,Z.
        """

        # Get psi at the specified R,Z
        psi = self.psiRZ(R,Z)

        # Get phiN at these psi
        phiN = self.phiNpsi(psi)

        return phiN

    def rhoTor_psi(self, psi):
        """
        Return rho toroidal = sqrt(phiN) for a given psi
        """

        # Get the normalised toroidal flux at the user-specified psi
        phiN = self.phiN_psi(psi)

        # Get rho toroidal
        rho_tor = np.sqrt(phiN)

        return rho_tor

    def rhoTorRZ(self, R, Z):
        """
        Return rho toroidal = sqrt(phiN) for a point at a given R,Z
        """

        # Get the normalised toroidal flux at the user-specified psi
        phiN = self.phiNRZ(R,Z)

        # Get rho toroidal
        rho_tor = np.sqrt(phiN)

        return rho_tor

    def rho_psi(self, R, Z):
        """
        Return rho = sqrt(phi/(pi*B0)) where B0 is the vacuum toroidal field on the geometric axis
        for a given psi. rho has units of length.
        """

        # Get the toroidal flux at the user-specifid psi
        phi = self.phi_psi(psi)

        # Get the vacuum toroidal field on the geometric axis
        B0 = self.fvac()/self.Rgeometric()
        
        # Get rho
        rho = np.sqrt(phi)/(np.pi*B0)

        return rho

    def rhoRZ(self, R, Z):
        """
        Return rho = sqrt(phi/(pi*B0)) where B0 is the vacuum toroidal field on the geometric axis
        for a point at a given R,Z. rho has units of length.
        """

        # Get psi at the user-specified R,Z
        psi = self.psiRZ(R,Z)

        # Get rho
        rho = self.rho_psi(psi)

        return rho

    def pprime(self, psinorm):
        """
        Return p' at given normalised psi
        """
        return self._profiles.pprime(psinorm)

    def ffprime(self, psinorm):
        """
        Return ff' at given normalised psi
        """
        return self._profiles.ffprime(psinorm)

    def JtorRZ(self, R, Z):
        """
        Return toridal current density Jtor at given R,Z
        """

        # Get psiN at the R,Z
        psiN = self.psiNRZ(R,Z)

        # Get the pprime and ffprime at these psiN
        pprime = self.pprime(psiN)
        ffprime = self.ffprime(psiN)

        # Calculate Jtor
        Jtor = R * pprime + ffprime / (mu0 * R)

        return Jtor

    def pressure(self, psinorm):
        """
        Returns plasma pressure at specified values of normalised psi
        """
        return self._profiles.pressure(psinorm)

    def separatrix(self, npoints=360):
        """
        Returns an array of npoints (R, Z) coordinates of the separatrix,
        equally spaced in geometric poloidal angle.
        """
        return array(critical.find_separatrix(self, ntheta=npoints, psi=self.psi()))[
            :, 0:2
        ]

    def psi_surfRZ(self, psiN=1.0, npoints=360):
        """
        Returns the R,Z of a flux surface specified by a value of psiN. This flux surface is closed on itself.
        """

        surf = critical.find_separatrix(self, opoint=None, xpoint=None, ntheta=npoints, psi=None, axis=None, psival=psiN)

        Rsurf = [point[0] for point in surf]
        Zsurf = [point[1] for point in surf]

        Rsurf.append(Rsurf[0])
        Zsurf.append(Zsurf[0])

        return np.array(Rsurf), np.array(Zsurf)

    def solve(self, profiles, Jtor=None, psi=None, psi_bndry=None):
        """
        Calculate the plasma equilibrium given new profiles
        replacing the current equilibrium.

        This performs the linear Grad-Shafranov solve

        profiles  - An object describing the plasma profiles.
                    At minimum this must have methods:
             .Jtor(R, Z, psi, psi_bndry)   -> [nx, ny]
             .pprime(psinorm)
             .ffprime(psinorm)
             .pressure(psinorm)
             .fpol(psinorm)

        Jtor : 2D array
            If supplied, specifies the toroidal current at each (R,Z) point
            If not supplied, Jtor is calculated from profiles by finding O,X-points

        psi_bndry  - Poloidal flux to use as the separatrix (plasma boundary)
                     If not given then X-point locations are used.
        """

        self._profiles = profiles
        self._updateBoundaryPsi()

        if Jtor is None:
            # Calculate toroidal current density
            if psi is None:
                psi = self.psi()
            Jtor = profiles.Jtor(self.R, self.Z, psi, psi_bndry=psi_bndry)

        # Set plasma boundary
        # Note that the Equilibrium is passed to the boundary function
        # since the boundary may need to run the G-S solver (von Hagenow's method)
        self._applyBoundary(self, Jtor, self.plasma_psi)

        # Right hand side of G-S equation
        rhs = -mu0 * self.R * Jtor

        # Copy boundary conditions
        rhs[0, :] = self.plasma_psi[0, :]
        rhs[:, 0] = self.plasma_psi[:, 0]
        rhs[-1, :] = self.plasma_psi[-1, :]
        rhs[:, -1] = self.plasma_psi[:, -1]

        # Call elliptic solver
        plasma_psi = self._solver(self.plasma_psi, rhs)

        self._updatePlasmaPsi(plasma_psi)

        # Update plasma current
        self._current = romb(romb(Jtor)) * self.dR * self.dZ
        self.Jtor = Jtor

    def _updateBoundaryPsi(self, psi=None):
        """
        For an input psi the magnetic axis and boundary psi are identified along
        with the core mask.

        Various logical checks occur, depending on whether or not the user
        wishes to check if the plasma is limited or not, as well as whether
        or not any xpoints are present.
        """

        if psi is None:
            psi = self.psi()

        opt, xpt = critical.find_critical(self.R, self.Z, psi)

        psi = psi

        if opt:
            # Magnetic axis flux taken as primary o-point flux
            self.psi_axis = opt[0][2]
            self.opt = opt[0]
            """
            Several options depending on if user wishes to check
            if the plasma becomes limited.
            """

            # The user wishes to check if the plasma is limited
            if self.check_limited and self.tokamak.wall:
                # A wall has actually been provided, proceed with checking

                # Obtain flux on machine limit points
                Rlimit = self.tokamak.limit_points_R
                Zlimit = self.tokamak.limit_points_Z

                """
                If an xpoint is present (plasma is potentially diverted)
                then we must remove any limit points above/below the
                primary xpoint as the PFR may land on these points,
                which would break the algorithm (at present) for extracting the boundary
                flux if the plasma were to infact be limited. There is a more advanced
                version of this alogorithm that is more robust that will be
                added in the future.
                """

                if xpt:
                    limit_args = np.ravel(
                        np.argwhere(abs(Zlimit) < abs(0.75 * xpt[0][1]))
                    )
                    Rlimit = Rlimit[limit_args]
                    Zlimit = Zlimit[limit_args]

                # Obtain the flux psi at these limiter points
                R = np.asarray(self.R[:, 0])
                Z = np.asarray(self.Z[0, :])

                # psi is transposed due to how FreeGS meshgrids R,Z
                psi_2d = interpolate.interp2d(x=R, y=Z, z=psi.T)

                # Get psi at the limit points
                psi_limit_points = np.zeros(len(Rlimit))
                for i in range(len(Rlimit)):
                    psi_limit_points[i] = psi_2d(Rlimit[i], Zlimit[i])[0]

                # Get index of maximum psi value
                indMax = np.argmax(psi_limit_points)

                # Extract R,Z of the contact point
                self.Rlim = Rlimit[indMax]
                self.Zlim = Zlimit[indMax]

                # Obtain maximum psi
                self.psi_limit = psi_limit_points[indMax]

                # Check if any xpoints are present
                if xpt:
                    # Get flux from the primary xpoint
                    self.psi_xpt = xpt[0][2]

                    # Choose between diverted or limited flux
                    self.psi_bndry = max(self.psi_xpt, self.psi_limit)

                    if self.psi_bndry == self.psi_limit:
                        self.is_limited = True

                    else:
                        self.is_limited = False

                    # Mask the core
                    self.mask = critical.core_mask(
                        self.R, self.Z, psi, opt, xpt, self.psi_bndry
                    )

                    # Use interpolation to find if a point is in the core.
                    self.mask_func = interpolate.RectBivariateSpline(
                        self.R[:, 0], self.Z[0, :], self.mask
                    )

                else:
                    # No xpoints, therefore psi_bndry = psi_limit
                    self.psi_bndry = self.psi_limit
                    self.is_limited = True
                    self.mask = None

            else:
                # Either a wall was not provided or the user did not wish to
                # check if the plasma was limited
                if xpt:
                    self.psi_xpt = xpt[0][2]
                    self.psi_bndry = self.psi_xpt
                    self.mask = critical.core_mask(self.R, self.Z, psi, opt, xpt)

                    # Use interpolation to find if a point is in the core.
                    self.mask_func = interpolate.RectBivariateSpline(
                        self.R[:, 0], self.Z[0, :], self.mask
                    )
                elif self._applyBoundary is fixedBoundary:
                    # No X-points, but using fixed boundary
                    self.psi_bndry = psi[0, 0]  # Value of psi on the boundary
                    self.mask = None  # All points are in the core region
                else:
                    self.psi_bndry = None
                    self.mask = None

                self.is_limited = False

    def _updatePlasmaPsi(self, plasma_psi):
        """
        Sets the plasma psi data, updates spline interpolation coefficients.
        Also updates:

        self.mask        2D (R,Z) array which is 1 in the core, 0 outside
        self.psi_axis    Value of psi on the magnetic axis
        self.psi_bndry   Value of psi on plasma boundary
        """
        self.plasma_psi = plasma_psi

        # Update spline interpolation
        self.psi_func = interpolate.RectBivariateSpline(
            self.R[:, 0], self.Z[0, :], plasma_psi
        )

        # Update the plasma axis and boundary flux as well as mask
        self._updateBoundaryPsi()

    def plot(self, axis=None, show=True, oxpoints=True):
        """
        Plot the equilibrium flux surfaces

        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning
        oxpoints - Plot X points as red circles, O points as green circles

        Returns
        -------

        axis  object from Matplotlib

        """
        from .plotting import plotEquilibrium

        return plotEquilibrium(self, axis=axis, show=show, oxpoints=oxpoints)

    def getForces(self):
        """
        Calculate forces on the coils

        Returns a dictionary of coil label -> force
        """
        return self.tokamak.getForces(self)

    def printForces(self):
        """
        Prints a table of forces on coils
        """
        print("Forces on coils")

        def print_forces(forces, prefix=""):
            for label, force in forces.items():
                if isinstance(force, dict):
                    print(prefix + label + " (circuit)")
                    print_forces(force, prefix=prefix + "  ")
                else:
                    print(
                        prefix
                        + label
                        + " : R = {0:.2f} kN , Z = {1:.2f} kN".format(
                            force[0] * 1e-3, force[1] * 1e-3
                        )
                    )

        print_forces(self.getForces())

    def innerOuterSeparatrix(self, Z=None):
        """
        Locate R coordinates of separatrix at the specified Z
        coordinate. Default Z will be the magnetic midplane.
        """

        if Z is None:
            Z = self.Zmagnetic()

        # Find the closest index to requested Z
        Zindex = np.argmin(abs(self.Z[0, :] - Z))

        # Normalised psi at this Z index
        psinorm = self.psiN()[:, Zindex]

        # Start from the magnetic axis
        Rindex_axis = np.argmin(abs(self.R[:, 0] - self.Rmagnetic()))

        # Inner separatrix
        # Get the maximum index where psi > 1 in the R index range from 0 to Rindex_axis
        outside_inds = np.argwhere(psinorm[:Rindex_axis] > 1.0)

        if outside_inds.size == 0:
            R_sep_in = self.Rmin
        else:
            Rindex_inner = np.amax(outside_inds)

            # Separatrix should now be between Rindex_inner and Rindex_inner+1
            # Linear interpolation
            R_sep_in = (
                self.R[Rindex_inner, Zindex] * (1.0 - psinorm[Rindex_inner + 1])
                + self.R[Rindex_inner + 1, Zindex] * (psinorm[Rindex_inner] - 1.0)
            ) / (psinorm[Rindex_inner] - psinorm[Rindex_inner + 1])

        # Outer separatrix
        # Find the minimum index where psi > 1
        outside_inds = np.argwhere(psinorm[Rindex_axis:] > 1.0)

        if outside_inds.size == 0:
            R_sep_out = self.Rmax
        else:
            Rindex_outer = np.amin(outside_inds) + Rindex_axis

            # Separatrix should now be between Rindex_outer-1 and Rindex_outer
            R_sep_out = (
                self.R[Rindex_outer, Zindex] * (1.0 - psinorm[Rindex_outer - 1])
                + self.R[Rindex_outer - 1, Zindex] * (psinorm[Rindex_outer] - 1.0)
            ) / (psinorm[Rindex_outer] - psinorm[Rindex_outer - 1])

        return R_sep_in, R_sep_out

    def intersectsWall(self):
        """Assess whether or not the core plasma touches the vessel
        walls. Returns True if it does intersect.
        """
        separatrix = self.separatrix()  # Array [:,2]
        wall = self.tokamak.wall  # Wall object with R and Z members (lists)

        return polygons.intersect(separatrix[:, 0], separatrix[:, 1], wall.R, wall.Z)

    def check_geometry(self, npoints=360):
        """Analyses the geometry of the core plasma, locating key points
        around the LCFS that define the shape of the core.
        """

        # Helper function that will (if required) insert the xpoint(s) into the
        # list of points that constitute the LCFS
        def insert_point_to_LCFS(Rp,Zp):

            distances = []

            # Calculate the distance between each point on the LCFS and the new point
            for Rlcfs_i, Zlcfs_i in zip(self.Rlcfs,self.Zlcfs):

                s = np.sqrt((Rlcfs_i - Rp)**2. + (Zlcfs_i - Zp)**2.)
                distances.append(s)

            indeces_sorted = np.argsort(distances)
            index_to_insert = max(indeces_sorted[0],indeces_sorted[1])

            self.Rlcfs = np.insert(self.Rlcfs,index_to_insert,Rp)
            self.Zlcfs = np.insert(self.Zlcfs,index_to_insert,Zp)

        # Get a large number of points along the LCFS
        separatrix = self.separatrix(npoints=npoints)  # Array [:,2]
        self.Rlcfs = [i[0] for i in separatrix]
        self.Zlcfs = [i[1] for i in separatrix]
        self.Rlcfs.append(self.Rlcfs[0])
        self.Zlcfs.append(self.Zlcfs[0])
        self.Rlcfs = np.asarray(self.Rlcfs)
        self.Zlcfs = np.asarray(self.Zlcfs)

        # Locate the extrema points in R, P1 and P3

        # The point of largest R, P1.

        ind_P1 = np.argmax(self.Rlcfs)
        self.R_P1 = self.Rlcfs[ind_P1]
        self.Z_P1 = self.Zlcfs[ind_P1]

        # The point of smallest R, P3.

        ind_P3 = np.argmin(self.Rlcfs)
        self.R_P3 = self.Rlcfs[ind_P3]
        self.Z_P3 = self.Zlcfs[ind_P3]

        # Get the xpoints
        self.opt, self.xpt = critical.find_critical(self.R, self.Z, self.psi())

        # Check if the plasma is diverted, and if so, what the configuration is
        if not self.is_limited:

            if len(self.xpt) == 1: # Only 1 xpoint, hence plasma must be SND

                self.DND = False # Cannot be DND if SND

            else: # More than 1 xpoint, can be either SND or DND

                # Check psi of primary and secondary separatrices. If they are
                # the same, the plasma is DND

                #if self.xpt[0][2] == self.xpt[1][2]: # DND
                if abs(self.xpt[1][2]-self.xpt[0][2])/self.xpt[0][2] < 1.0e-04: # DND

                    self.DND = True
                    self.LSND = False
                    self.USND = False

                    # Insert first and second xpoints into the LCFS
                    insert_point_to_LCFS(self.xpt[0][0],self.xpt[0][1])
                    insert_point_to_LCFS(self.xpt[1][0],self.xpt[1][1])

                else: # not DND, ie. SND

                    self.DND = False

                    # Insert primary xpoint into the LCFS
                    insert_point_to_LCFS(self.xpt[0][0],self.xpt[0][1])

            if not self.DND: # If the plasma is not DND, check if it is LSND or USND

                # Check xpoint Z position to determine if USND or LSND

                if self.xpt[0][1] < 0.0: # LSND

                    self.LSND = True
                    self.USND = False

                else: # USND

                    self.LSND = False
                    self.USND = True

        else: # limited

            self.DND = False
            self.LSND = False
            self.USND = False

        # Make a Shapely LineString of the LCFS. This will be used
        # to find the squareness of the plasma.
        lcfs = LineString(zip(self.Rlcfs,self.Zlcfs))

        # P2 will be the upper midplane LCFS extrema

        ind_P2 = np.argmax(self.Zlcfs)
        self.R_P2 = self.Rlcfs[ind_P2]
        self.Z_P2 = self.Zlcfs[ind_P2]

        # P4 will be the lower midplane LCFS extrema

        ind_P4 = np.argmin(self.Zlcfs)
        self.R_P4 = self.Rlcfs[ind_P4]
        self.Z_P4 = self.Zlcfs[ind_P4]

        # Next, we determine 4 more points, P5,6,7,8 that are used to
        # defined the squareness of the plasma for the upper outer quadrant.

        # Get a = Z difference between P1 and P2
        a = self.Z_P2 - self.Z_P1

        # Get b = R difference between P1 and P2
        b = self.R_P1 - self.R_P2

        # Get the elipse intersection point, P5
        self.R_P5 = self.R_P2 + b*np.sqrt(0.5)
        self.Z_P5 = self.Z_P1 + a*np.sqrt(0.5)

        # Define P6, a point at the R of P1 and the Z of P2
        self.R_P6 = self.R_P1
        self.Z_P6 = self.Z_P2

        # Define P7, a point at the R of P2 and the Z of P1
        self.R_P7 = self.R_P2
        self.Z_P7 = self.Z_P1

        # Get P8, the intersection point of the LCFS and a line joining P6 to P7
        line = LineString(zip([self.R_P6,self.R_P7],[self.Z_P6,self.Z_P7]))
        intersection_point = intersection(lcfs,line)
        
        self.R_P8 = intersection_point.x
        self.Z_P8 = intersection_point.y

        # Next, we determine 4 more points, P9,10,11,12 that are used to
        # defined the squareness of the plasma for the lower outer quadrant.
        # Get a = Z difference between P1 and P4
        a = self.Z_P1 - self.Z_P4

        # Get b = R difference between P1 and P4
        b = self.R_P1 - self.R_P4

        # Get the elipse intersection point, P9
        self.R_P9 = self.R_P4 + b*np.sqrt(0.5)
        self.Z_P9 = self.Z_P1 - a*np.sqrt(0.5)

        # Define P10, a point at the R of P1 and the Z of P4
        self.R_P10 = self.R_P1
        self.Z_P10 = self.Z_P4

        # Define P11, a point at the R of P4 and the Z of P1
        self.R_P11 = self.R_P4
        self.Z_P11 = self.Z_P1

        # Get P12, the intersection point of the LCFS and a line joining P10 to P11
        line = LineString(zip([self.R_P10,self.R_P11],[self.Z_P10,self.Z_P11]))
        intersection_point = intersection(lcfs,line)

        self.R_P12 = intersection_point.x
        self.Z_P12 = intersection_point.y

        # Next, we determine 4 more points, P13,14,15,16 that are used to
        # defined the squareness of the plasma for the upper inner quadrant.
        # Get a = Z difference between P2 and P3
        a = self.Z_P2 - self.Z_P3

        # Get b = R difference between P2 and P3
        b = self.R_P2 - self.R_P3 ##

        # Get the elipse intersection point, P13
        self.R_P13 = self.R_P2 - b*np.sqrt(0.5)
        self.Z_P13 = self.Z_P3 + a*np.sqrt(0.5)

        # Define P14, a point at the R of P3 and the Z of P2
        self.R_P14 = self.R_P3
        self.Z_P14 = self.Z_P2

        # Define P15, a point at the R of P2 and the Z of P3
        self.R_P15 = self.R_P2
        self.Z_P15 = self.Z_P3

        # Get P16, the intersection point of the LCFS and a line joining P14 to P15
        line = LineString(zip([self.R_P14,self.R_P15],[self.Z_P14,self.Z_P15]))
        intersection_point = intersection(lcfs,line)

        self.R_P16 = intersection_point.x
        self.Z_P16 = intersection_point.y

        # Next, we determine 4 more points, P17,18,19,20 that are used to
        # defined the squareness of the plasma for the lower inner quadrant.
        # Get a = Z difference between P4 and P3
        a = self.Z_P3 - self.Z_P4

        # Get b = R difference between P4 and P3
        b = self.R_P4 - self.R_P3 ##

        # Get the elipse intersection point, P17
        self.R_P17 = self.R_P4 - b*np.sqrt(0.5)
        self.Z_P17 = self.Z_P3 - a*np.sqrt(0.5)

        # Define P18, a point at the R of P3 and the Z of P4
        self.R_P18 = self.R_P3
        self.Z_P18 = self.Z_P4

        # Define P19, a point at the R of P4 and the Z of P3
        self.R_P19 = self.R_P4
        self.Z_P19 = self.Z_P3

        # Get P20, the intersection point of the LCFS and a line joining P18 to P19
        line = LineString(zip([self.R_P18,self.R_P19],[self.Z_P18,self.Z_P19]))
        intersection_point = intersection(lcfs,line)

        self.R_P20 = intersection_point.x
        self.Z_P20 = intersection_point.y

        # Plot these key geometry points - useful for debugging
        '''fig, ax = plt.subplots()
        ax.plot(self.Rlcfs,self.Zlcfs,'r')
        ax.scatter(self.R_P1,self.Z_P1,marker='x',label='P1')
        ax.scatter(self.R_P2,self.Z_P2,marker='x',label='P2')
        ax.scatter(self.R_P3,self.Z_P3,marker='x',label='P3')
        ax.scatter(self.R_P4,self.Z_P4,marker='x',label='P4')
        ax.scatter(self.R_P5,self.Z_P5,marker='x',label='P5')
        ax.scatter(self.R_P6,self.Z_P6,marker='x',label='P6')
        ax.scatter(self.R_P7,self.Z_P7,marker='x',label='P7')
        ax.scatter(self.R_P8,self.Z_P8,marker='x',label='P8')
        ax.scatter(self.R_P9,self.Z_P9,marker='x',label='P9')
        ax.scatter(self.R_P10,self.Z_P10,marker='x',label='P10')
        ax.scatter(self.R_P11,self.Z_P11,marker='x',label='P11')
        ax.scatter(self.R_P12,self.Z_P12,marker='x',label='P12')
        ax.scatter(self.R_P11,self.Z_P11,marker='x',label='P11')
        ax.scatter(self.R_P12,self.Z_P12,marker='x',label='P12')
        ax.scatter(self.R_P13,self.Z_P13,marker='x',label='P13')
        ax.scatter(self.R_P14,self.Z_P14,marker='x',label='P14')
        ax.scatter(self.R_P15,self.Z_P15,marker='x',label='P15')
        ax.scatter(self.R_P16,self.Z_P16,marker='x',label='P16')
        ax.scatter(self.R_P17,self.Z_P17,marker='x',label='P17')
        ax.scatter(self.R_P18,self.Z_P18,marker='x',label='P18')
        ax.scatter(self.R_P19,self.Z_P19,marker='x',label='P19')
        ax.scatter(self.R_P20,self.Z_P20,marker='x',label='P20')
        ax.plot([self.R_P6,self.R_P7],[self.Z_P6,self.Z_P7],'k')
        ax.plot([self.R_P10,self.R_P11],[self.Z_P10,self.Z_P11],'k')
        ax.plot([self.R_P14,self.R_P15],[self.Z_P14,self.Z_P15],'k')
        ax.plot([self.R_P18,self.R_P19],[self.Z_P18,self.Z_P19],'k')
        ax.scatter(self.Rgeometric(),self.Zgeometric(),color='tab:orange',marker='+',label=r'$(R_{geo},Z_{geo})$')
        ax.scatter(self.Rcentroid(),self.Zcentroid(),color='b',marker='+',label=r'$(R_{cent},Z_{cent})$')
        ax.scatter(self.Rmagnetic(),self.Zmagnetic(),color='g',marker='+',label=r'$(R_{mag},Z_{mag})$')
        ax.set_aspect('equal')
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        #ax.legend(loc='lower right')
        plt.show()
        '''

    def magneticAxis(self):
        """Returns the location of the magnetic axis as a list [R,Z,psi]"""

        return self.opt[0]

    def Rmagnetic(self):
        """The major radius R of magnetic major radius"""
        return self.magneticAxis()[0]

    def Zmagnetic(self):
        """The height Z of magnetic axis"""
        return self.magneticAxis()[1]

    def geometricAxis(self):
        """Returns the geometric axis of the plasma [Rgeo,Zgeo]
        
        Definitions for plasma boundary shape characterisation taken from:

        Luce, T.C., 2013. An analytic functional form for characterization and generation
        of axisymmetric plasma boundaries. Plasma physics and controlled fusion, 55(9), p.095009.

        DOI 10.1088/0741-3335/55/9/095009
        """

        Rgeo = 0.5*(self.R_P3 + self.R_P1)
        Zgeo = 0.5*(self.Z_P4 + self.Z_P2)

        return [Rgeo, Zgeo]

    def Rgeometric(self):
        """Locates major radius R of the geometric axis."""

        Rgeo = 0.5*(self.R_P3 + self.R_P1)

        return Rgeo

    def Zgeometric(self):
        """Locates the height Z of the geometric axis."""

        Zgeo = 0.5*(self.Z_P4 + self.Z_P2)

        return Zgeo

    def centroidAxis(self):
        """Returns the geometric centroid of the LCFS [Rcent,Zcent]"""

        # Get a list of [r,z] points of the LCFS
        points = [[r,z] for r,z in zip(self.Rlcfs,self.Zlcfs)]

        # Make a polygon of the LCFS
        P = Polygon(points)

        # Find the centroid of this polygon
        centroid = P.centroid

        return [centroid.x,centroid.y]

    def Rcentroid(self):
        """Locates najor radius R of the LCFS centroid."""
        return self.centroidAxis()[0]

    def Zcentroid(self):
        """Locates height z of the LCFS centroid."""
        return self.centroidAxis()[1]

    def minorRadius(self):
        """Calculates minor radius of the plasma, a."""

        a = 0.5*(self.R_P1 - self.R_P3)

        return a

    def aspectRatio(self):
        """Calculates the plasma aspect ratio, A.

        A = R0/a where R0 = major radius, a = minor radius.
        """

        A = self.Rgeometric() / self.minorRadius()

        return A

    def inverseAspectRatio(self):
        """Calculates inverse of the aspect ratio.

        epsilon = 1/A
        A = R0/a where R0 = major radius, a = minor radius.
        """

        epsilon = 1.0/self.aspectRatio()

        return epsilon

    def elongation(self):
        """Calculates plasma elongation, kappa."""

        kappa = (self.Z_P2 - self.Z_P4) / (2. * self.minorRadius())

        return kappa

    def elongationUpper(self):
        """Calculates the upper elongation, kappa_u, of the plasma."""

        kappa_u = (self.Z_P2 - self.Z_P1) / self.minorRadius()

        return kappa_u

    def elongationLower(self):
        """Calculates the lower elongation, kappa_l, of the plasma."""

        kappa_l = (self.Z_P1 - self.Z_P4) / self.minorRadius()

        return kappa_l

    def effectiveElongation(self):
        """Calculates plasma effective elongation, kappa_a, using the plasma volume.
        
        kappa_a = V / (2 * pi^2 * Rgeo * aminor)
        """

        kappa_a = self.plasmaVolume() / (
            2.0
            * np.pi
            * self.Rgeometric()
            * np.pi
            * self.minorRadius() ** 2
        )

        return kappa_a

    def triangularityUpper(self):
        """Calculates the upper triangularity, delta_u, of the plasma.
        """

        delta_u = (self.Rgeometric() - self.R_P2) / self.minorRadius()

        return delta_u

    def triangularityLower(self):
        """Calculates the lower triangularity, delta_l, of the plasma.
        """

        delta_l = (self.Rgeometric() - self.R_P4) / self.minorRadius()

        return delta_l

    def triangularity(self):
        """Calculates plasma triangularity, delta.

        Here delta is defined as the average of the upper
        and lower triangularities.
        """

        delta_u = self.triangularityUpper()
        delta_l = self.triangularityLower()

        delta = 0.5 * (delta_u + delta_l)

        return delta

    def squarenessUpperOuter(self):
        """Calculates the upper outer quadrant plasma squareness, zeta_uo."""

        # Calculate distance between P7 and P8
        dist_P7_8 = np.sqrt((self.R_P7-self.R_P8)**2. + (self.Z_P7-self.Z_P8)**2.)

        # Calculate distance between P7 and P5
        dist_P7_5 = np.sqrt((self.R_P7-self.R_P5)**2. + (self.Z_P7-self.Z_P5)**2.)

        # Calculate distance between P5 and P6
        dist_P5_6 = np.sqrt((self.R_P5-self.R_P6)**2. + (self.Z_P5-self.Z_P6)**2.)

        zeta_uo = (dist_P7_8 - dist_P7_5) / dist_P5_6

        return zeta_uo

    def squarenessLowerOuter(self):
        """Calculates the lower outer quadrant plasma squareness, zeta_lo."""

        # Calculate distance between P11 and P12
        dist_P11_12 = np.sqrt((self.R_P11-self.R_P12)**2. + (self.Z_P11-self.Z_P12)**2.)

        # Calculate distance between P11 and P9
        dist_P11_9 = np.sqrt((self.R_P11-self.R_P9)**2. + (self.Z_P11-self.Z_P9)**2.)

        # Calculate distance between P9 and P10
        dist_P9_10 = np.sqrt((self.R_P9-self.R_P10)**2. + (self.Z_P9-self.Z_P10)**2.)

        zeta_lo = (dist_P11_12 - dist_P11_9) / dist_P9_10

        return zeta_lo

    def squarenessUpperInner(self):
        """Calculates the upper inner quadrant plasma squareness, zeta_ui."""

        # Calculate distance between P15 and P16
        dist_P15_16 = np.sqrt((self.R_P15-self.R_P16)**2. + (self.Z_P15-self.Z_P16)**2.)

        # Calculate distance between P15 and P13
        dist_P15_13 = np.sqrt((self.R_P15-self.R_P13)**2. + (self.Z_P15-self.Z_P13)**2.)

        # Calculate distance between P13 and P14
        dist_P13_14 = np.sqrt((self.R_P13-self.R_P14)**2. + (self.Z_P13-self.Z_P14)**2.)

        zeta_ui = (dist_P15_16 - dist_P15_13) / dist_P13_14

        return zeta_ui

    def squarenessLowerInner(self):
        """Calculates the lower inner quadrant plasma squareness, zeta_li."""

        # Calculate distance between P19 and P20
        dist_P19_20= np.sqrt((self.R_P19-self.R_P20)**2. + (self.Z_P19-self.Z_P20)**2.)

        # Calculate distance between P19 and P17
        dist_P19_17 = np.sqrt((self.R_P19-self.R_P17)**2. + (self.Z_P19-self.Z_P17)**2.)

        # Calculate distance between P17 and P18
        dist_P17_18 = np.sqrt((self.R_P17-self.R_P18)**2. + (self.Z_P17-self.Z_P18)**2.)

        zeta_li = (dist_P19_20 - dist_P19_17) / dist_P17_18
        return zeta_li

    def calc_miller_params(self,show=False):
        """ Fits a shape parameterised by the
        Miller eXtended Harmonic (MXH) model to the LCFS taken from
        
        Arbon, R., Candy, J. and Belli, E.A., 2020. Rapidly-convergent
        flux-surface shape parameterization. Plasma Physics and Controlled Fusion, 63(1), p.012001.

        DOI 10.1088/1361-6587/abc63b.

        Adapated from original code provided by Michail Anastopoulos.
        """

        # R of the boundary from the Miller representation
        def R_MXH(c,R0,rho0,th):
            th_mxh = th + c[0] + c[1]*np.cos(th) + c[2]*np.sin(th) + c[3]*np.cos(2*th) + c[4]*np.sin(2*th) \
            + c[5]*np.cos(3*th) + c[6]*np.sin(3*th) + c[7]*np.cos(4*th) + c[8]*np.sin(4*th)	
            R = R0 + rho0 * np.cos( th_mxh )
            return R
        
        # Residual in R from fitting the Miller boundary to the LCFS
        def residual(c,Rb,R0,rho0,th):
            return Rb - R_MXH(c,R0,rho0,th)

        # Number of points constituting the LCFS
        nbndry = len(self.Rlcfs)

        # Obtain the major radius of the geometric axis
        R0 = self.Rgeometric()

        # Obtain the height of the geometric axis
        Z0 = self.Zgeometric()
        
        # Obtain the elongation
        kappa = self.elongation()

        # Obtain the minor radius
        rho0 = self.minorRadius()

        # Calculate the poloidal angle of each boundary point
        th = []
        for i in range(nbndry):
            if (self.Zlcfs[i]-Z0)/(kappa*rho0)>1.0:
                th.append(np.arcsin(1.0))
            elif (self.Zlcfs[i]-Z0)/(kappa*rho0)<-1.0:
                th.append(np.arcsin(-1.0))
            else:
                th.append(np.arcsin((self.Zlcfs[i]-Z0)/(kappa*rho0)))
        th = np.array(th)
        for i in range(1,nbndry):
            if th[i]<th[i-1]:
                th[i] = np.pi-th[i]
            if th[i]<th[i-1]:
                th[i] = 3*np.pi-th[i]
        th -= np.pi 
        th = -th

        # Fit a Miller parameterised boundary to the LCFS
        c0 = np.zeros([9])
        c, flags = leastsq(residual, c0, args=(self.Rlcfs,R0,rho0,th))

        # Record these miller parameters
        self.miller_params = c

        # Plot this Miller boundary
        if show:

            x = np.zeros([nbndry])
            z = np.zeros([nbndry])
            th = np.linspace(-np.pi,np.pi,nbndry)
            th_mxh = th + c[0] + c[1]*np.cos(th) + c[2]*np.sin(th) + c[3]*np.cos(2*th) + c[4]*np.sin(2*th) + \
            c[5]*np.cos(3*th) + c[6]*np.sin(3*th) + c[7]*np.cos(4*th) + c[8]*np.sin(4*th)	
            for i in range(nbndry):
                x[i] = R0 + rho0 * np.cos( th_mxh[i] )
                z[i] = Z0 + kappa * rho0 * np.sin(th[i])

            fig, ax = plt.subplots()
            ax.plot(x,z,label='MXH')
            ax.plot(self.Rlcfs,self.Zlcfs,'r',label='LCFS')
            ax.set_aspect('equal')
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')
            ax.legend()
            plt.show()

    def flux_surface_averaged_Bpol2(self, psiN=1.0, npoints=1000):
        """
        Calculates the flux surface averaged value of the square of the poloidal field.
        """

        # Get R, Z points of the flux surface
        Rsurf, Zsurf = self.psi_surfRZ(psiN=psiN,npoints=npoints)

        self.Rsurf = Rsurf
        self.Zsurf = Zsurf

        # Get the poloidal field
        Bpol_surf = self.Bpol(Rsurf,Zsurf)

        # Get the square of the poloidal field
        Bpol_surf2 = Bpol_surf**2.0

        # Get dl along the surface
        dl = np.sqrt(np.diff(Rsurf)**2.0 + np.diff(Zsurf)**2.0)
        dl = np.insert(dl,0,0.0)

        # Get l along the surface
        l = np.cumsum(dl)

        # Calculate the flux surface averaged quantity
        return np.trapz(x=l, y=Bpol_surf2 * Bpol_surf) / np.trapz(x=l, y=np.ones(np.size(l)) * Bpol_surf)

    def shafranovShift(self):
        """Calculates the plasma shafranov shift
        [delta_shafR,delta_shafZ] where

        delta_shafR = Rmagnetic - Rgeo
        delta_shafR = Zmagnetic - z0
        """

        Rmag = self.Rmagnetic()
        Zmag = self.Zmagnetic()

        Rgeo = self.Rgeometric()
        z0 = self.Zgeometric()

        return np.array([Rmag - Rgeo, Zmag - z0])

    def internalInductance1(self):
        """Calculates li1 plasma internal inductance.
        
        li1 = ( (1 + kappa^2) / (2 * kappa_a) ) * ( (2 * V * <Bpol^2>) / ( (mu0 Ip)^2 * Rgeo) )
        """

        R = self.R
        Z = self.Z
        # Produce array of Bpol^2 in (R,Z)
        B_polvals_2 = self.Bz(R, Z) ** 2 + self.Br(R, Z) ** 2

        volume_averaged_Bp2 = self.calc_volume_averaged(B_polvals_2)

        vol = self.plasmaVolume()
        Ip = self.plasmaCurrent()
        R_geo = self.Rgeometric()
        kappa = self.elongation()
        kappa_a = self.effectiveElongation()

        return ((1.+kappa*kappa)/(2.*kappa_a)) * (2.0*vol*volume_averaged_Bp2) / ((mu0*Ip)**2.0 * R_geo)

    def internalInductance2(self):
        """Calculates li2 plasma internal inductance.
        
        li2 = (2 * V * <Bpol^2>) / ( (mu0 Ip)^2 * Rmag)
        """

        R = self.R
        Z = self.Z
        # Produce array of Bpol^2 in (R,Z)
        B_polvals_2 = self.Bz(R, Z) ** 2 + self.Br(R, Z) ** 2

        volume_averaged_Bp2 = self.calc_volume_averaged(B_polvals_2)
        
        vol = self.plasmaVolume()
        Ip = self.plasmaCurrent()
        R_mag = self.Rmagnetic()

        return (2.0*vol*volume_averaged_Bp2) / ((mu0*Ip)**2.0 * R_mag)

    def internalInductance3(self):
        """Calculates li3 plasma internal inductance.
        
        li3 = (2 * V * <Bpol^2>) / ( (mu0 Ip)^2 * Rgeo)
        """

        R = self.R
        Z = self.Z
        # Produce array of Bpol^2 in (R,Z)
        B_polvals_2 = self.Bz(R, Z) ** 2 + self.Br(R, Z) ** 2

        volume_averaged_Bp2 = self.calc_volume_averaged(B_polvals_2)

        vol = self.plasmaVolume()
        Ip = self.plasmaCurrent()
        R_geo = self.Rgeometric()

        return (2.0*vol*volume_averaged_Bp2) / ((mu0*Ip)**2.0 * R_geo)

    def internalInductance(self):
        """Calculates the full plasma internal inductance Li

        (1/2) Li * Ip^2 = int( (Bpol^2 / (2mu0) ) dV )
        """

        # Produce array of Bpol in (R,Z)
        B_polvals_2 = self.Bpol(self.R, self.Z) ** 2

        # Volume integrated Bpol^2
        Bpol2_integral = self.calc_volume_integrated(B_polvals_2)

        # Get the plasma current
        Ip = self.plasmaCurrent()

        return Bpol2_integral / (mu0 * Ip * Ip)

    def poloidalBeta(self):
        """Return the poloidal beta.
        
        betaP = 2 * mu0 * <p> / <<Bpol^2>>
        """

        # Normalised psi
        psi_norm = self.psiN()

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        volume_averaged_pressure = self.calc_volume_averaged(pressure)
        
        line_averaged_Bpol2_lcfs = self.flux_surface_averaged_Bpol2(psiN=1.0)

        return (2.0 * mu0 * volume_averaged_pressure) / line_averaged_Bpol2_lcfs

    def poloidalBeta2(self):
        """Return the poloidal beta.

        betaP2 = (4 * V * <p>) / (mu0 * Ip^2 * Raxis)
        """

        # Normalised psi
        psi_norm = self.psiN()

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        volume_averaged_pressure = self.calc_volume_averaged(pressure)

        return (4 * self.plasmaVolume() * volume_averaged_pressure) / (mu0 * self.plasmaCurrent()**2.0 * self.Rmagnetic())

    def poloidalBeta3(self):
        """Return the poloidal beta.

        betaP3 = (4 * V * <p>) / (mu0 * Ip^2 * Rgeo)
        """

        # Normalised psi
        psi_norm = self.psiN()

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        volume_averaged_pressure = self.calc_volume_averaged(pressure)

        return (4 * self.plasmaVolume() * volume_averaged_pressure) / (mu0 * self.plasmaCurrent()**2.0 * self.Rgeometric())

    def toroidalBeta(self):
        """Calculate plasma toroidal beta (not a percentage)."""

        # Normalised psi
        psi_norm = self.psiN()

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        volume_averaged_pressure = self.calc_volume_averaged(pressure)

        # Toroidal field at geometric axis
        geo = self.geometricAxis()
        Bt = self.Btor(geo[0], geo[1])
        
        return 2.0*mu0*volume_averaged_pressure/Bt**2.0

    def totalBeta(self):
        """Calculate plasma total beta"""
        return 1.0 / ((1.0 / self.poloidalBeta()) + (1.0 / self.toroidalBeta()))

    def betaN(self):
        """Calculate normalised plasma beta (not a percentage)."""

        # Toroidal field at geometric axis
        geo = self.geometricAxis()
        Bt = self.Btor(geo[0], geo[1])

        return (
              1.0e06
            * self.totalBeta()
            * ((self.minorRadius() * Bt) / (self.plasmaCurrent()))
        )

    def pressure_ave(self):
        """Calculate volume averaged pressure, Pa."""

        # Normalised psi
        psi_norm = self.psiN()

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        return self.calc_volume_averaged(pressure)

    def w_th(self):
        """
        Stored thermal energy in plasma, J.

        Wth = 3/2 * int(p dV)
        """

        # Normalised psi
        psi_norm = self.psiN()

        # Plasma pressure
        pressure = self.pressure(psi_norm)

        pressure_integral = self.calc_volume_integrated(pressure)
        thermal_energy = (3.0 / 2.0) * pressure_integral

        return thermal_energy

    def qcyl(self):
        """
        Cylindrical safety factor.
        """

        eps = self.inverseAspectRatio()
        a = self.minorRadius()

        btor = self.fvac() / self.Rgeometric()
        Ip = self.plasmaCurrent()

        kappa = self.elongation()

        val = 0.5 * (1 + kappa * kappa) * ((2.0 * np.pi * a * eps * btor) / (mu0 * Ip))

        return val

    def calc_volume_integrated(self,field):
        """
        Calculates the volume integral of the input field.
        """

        dV = 2.0 * np.pi * self.R * self.dR  *self.dZ

        if self.mask is not None:  # Only include points in the core
            dV *= self.mask

        return romb(romb(field * dV))

    def calc_volume_averaged(self,field):
        """
        Calculates the volume average of the input field.
        """

        volume_integrated_field = self.calc_volume_integrated(field)
        
        return volume_integrated_field / self.plasmaVolume()

    def currentAxis(self):
        """
        Returns the location of the current centre as a list [R,Z]
        
        Rcur = 
        """

        Ip = romb(romb(self.Jtor)) * self.dR * self.dZ

        Rcur = (1./Ip) * romb(romb(self.R * self.Jtor)) * self.dR * self.dZ
        Zcur = (1./Ip) * romb(romb(self.Z * self.Jtor)) * self.dR * self.dZ

        return [Rcur,Zcur]

    def Rcurrent(self):
        """Returns the R coordinate of the current centre"""

        Rcur, _ = self.currentAxis()
        return Rcur

    def Zcurrent(self):
        """Returns the Z coordinate of the current centre"""

        _, Zcur = self.currentAxis()
        return Zcur

    def decay_index(self):
        """
        Calculate the decay index n = -(R/Bz) * (dBz/dR).
        This is calculated on the magnetic axis.
        """

        R = self.R
        Z = self.Z
        Bz = self.Bz(R,Z)

        Bz_func = interpolate.RectBivariateSpline(
            self.R[:, 0], self.Z[0, :], Bz
        )

        Rmag = self.Rmagnetic()
        Zmag = self.Zmagnetic()
        Bz_mag = self.Bz(Rmag,Zmag)
        dBz_dR_mag = Bz_func(Rmag, Zmag, dx=1, grid=False)

        return -(Rmag/Bz_mag) * dBz_dR_mag

def refine(eq, nx=None, ny=None):
    """
    Double grid resolution, returning a new equilibrium


    """
    # Interpolate the plasma psi
    # plasma_psi = multigrid.interpolate(eq.plasma_psi)
    # nx, ny = plasma_psi.shape

    # By default double the number of intervals
    if not nx:
        nx = 2 * (eq.R.shape[0] - 1) + 1
    if not ny:
        ny = 2 * (eq.R.shape[1] - 1) + 1

    result = Equilibrium(
        tokamak=eq.tokamak,
        Rmin=eq.Rmin,
        Rmax=eq.Rmax,
        Zmin=eq.Zmin,
        Zmax=eq.Zmax,
        boundary=eq._applyBoundary,
        order=eq.order,
        nx=nx,
        ny=ny,
    )

    plasma_psi = eq.psi_func(result.R, result.Z, grid=False)

    result._updatePlasmaPsi(plasma_psi)

    if hasattr(eq, "_profiles"):
        result._profiles = eq._profiles

    if hasattr(eq, "control"):
        result.control = eq.control

    return result

def coarsen(eq):
    """
    Reduce grid resolution, returning a new equilibrium
    """
    plasma_psi = multigrid.restrict(eq.plasma_psi)
    nx, ny = plasma_psi.shape

    result = Equilibrium(
        tokamak=eq.tokamak,
        Rmin=eq.Rmin,
        Rmax=eq.Rmax,
        Zmin=eq.Zmin,
        Zmax=eq.Zmax,
        nx=nx,
        ny=ny,
    )

    result._updatePlasmaPsi(plasma_psi)

    if hasattr(eq, "_profiles"):
        result._profiles = eq._profiles

    if hasattr(eq, "control"):
        result.control = eq.control

    return result


def newDomain(eq, Rmin=None, Rmax=None, Zmin=None, Zmax=None, nx=None, ny=None):
    """Creates a new Equilibrium, solving in a different domain.
    The domain size (Rmin, Rmax, Zmin, Zmax) and resolution (nx,ny)
    are taken from the input equilibrium eq if not specified.
    """
    if Rmin is None:
        Rmin = eq.Rmin
    if Rmax is None:
        Rmax = eq.Rmax
    if Zmin is None:
        Zmin = eq.Zmin
    if Zmax is None:
        Zmax = eq.Zmax
    if nx is None:
        nx = eq.R.shape[0]
    if ny is None:
        ny = eq.R.shape[0]

    # Create a new equilibrium with the new domain
    result = Equilibrium(
        tokamak=eq.tokamak, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=nx, ny=ny
    )

    # Calculate the current on the old grid
    profiles = eq._profiles
    Jtor = profiles.Jtor(eq.R, eq.Z, eq.psi(), eq.psi_bndry)

    # Interpolate Jtor onto new grid
    Jtor_func = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], Jtor)
    Jtor_new = Jtor_func(result.R, result.Z, grid=False)

    result._applyBoundary(result, Jtor_new, result.plasma_psi)

    # Right hand side of G-S equation
    rhs = -mu0 * result.R * Jtor_new

    # Copy boundary conditions
    rhs[0, :] = result.plasma_psi[0, :]
    rhs[:, 0] = result.plasma_psi[:, 0]
    rhs[-1, :] = result.plasma_psi[-1, :]
    rhs[:, -1] = result.plasma_psi[:, -1]

    # Call elliptic solver
    plasma_psi = result._solver(result.plasma_psi, rhs)

    result._updatePlasmaPsi(plasma_psi)

    # Solve once more, calculating Jtor using new psi
    result.solve(profiles)

    return result


if __name__ == "__main__":

    # Test the different spline interpolation routines

    from numpy import ravel
    import matplotlib.pyplot as plt

    import machine

    tokamak = machine.TestTokamak()

    Rmin = 0.1
    Rmax = 2.0

    eq = Equilibrium(tokamak, Rmin=Rmin, Rmax=Rmax)

    import constraints

    xpoints = [(1.2, -0.8), (1.2, 0.8)]
    constraints.xpointConstrain(eq, xpoints)

    psi = eq.psi()

    tck = interpolate.bisplrep(ravel(eq.R), ravel(eq.Z), ravel(psi))
    spline = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psi)
    f = interpolate.interp2d(eq.R[:, 0], eq.Z[0, :], psi, kind="cubic")

    plt.plot(eq.R[:, 10], psi[:, 10], "o")

    r = linspace(Rmin, Rmax, 1000)
    z = eq.Z[0, 10]
    plt.plot(r, f(r, z), label="f")

    plt.plot(r, spline(r, z), label="spline")

    plt.plot(r, interpolate.bisplev(r, z, tck), label="bisplev")

    plt.legend()
    plt.show()
