"""
Plasma control system

Use constraints to adjust coil currents
"""

from numpy import dot, transpose, eye, array, inf
from numpy.linalg import inv, norm
import numpy as np
from scipy import optimize
from . import critical
import matplotlib.pyplot as plt

class constrain(object):
    """
    Adjust coil currents using constraints. To use this class,
    first create an instance by specifying the constraints

    >>> controlsystem = constrain(xpoints = [(1.0, 1.1), (1.0,-1.0)])

    controlsystem will now attempt to create x-points at
    (R,Z) = (1.0, 1.1) and (1.0, -1.1) in any Equilibrium

    >>> controlsystem(eq)

    where eq is an Equilibrium object which is modified by
    the call.

    The constraints which can be set are:

    xpoints - A list of X-point (R,Z) locations

    isoflux - A list of tuples (R1,Z1, R2,Z2)

    psivals - A list of (R,Z,psi) values

    At least one of the above constraints must be included.

    gamma - A scalar, minimises the magnitude of the coil currents

    The following constraitns are entirely optional:

    current_lims - A list of tuples [(l1,u1),(l2,u2)...(lN,uN)] for the upper
    and lower bounds on the currents in each coil.

    max_total_current - The maximum total current through the coilset.
    """

    def __init__(
        self,
        xpoints=[],
        gamma=1e-12,
        isoflux=[],
        psivals=[],
        current_lims=None,
        max_total_current=None,
    ):
        """
        Create an instance, specifying the constraints to apply
        """
        self.xpoints = xpoints
        self.gamma = gamma
        self.isoflux = isoflux
        self.psivals = psivals
        self.current_lims = current_lims
        self.max_total_current = max_total_current

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()

        constraint_matrix = []
        constraint_rhs = []
        for xpt in self.xpoints:
            # Each x-point introduces two constraints
            # 1) Br = 0

            Br = eq.Br(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Br)
            constraint_matrix.append(tokamak.controlBr(xpt[0], xpt[1]))

            # 2) Bz = 0

            Bz = eq.Bz(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Bz)
            constraint_matrix.append(tokamak.controlBz(xpt[0], xpt[1]))

        # Constrain points to have the same flux
        for r1, z1, r2, z2 in self.isoflux:
            # Get Psi at (r1,z1) and (r2,z2)
            p1 = eq.psiRZ(r1, z1)
            p2 = eq.psiRZ(r2, z2)
            constraint_rhs.append(p2 - p1)

            # Coil responses
            c1 = tokamak.controlPsi(r1, z1)
            c2 = tokamak.controlPsi(r2, z2)
            # Control for the difference between p1 and p2
            c = [c1val - c2val for c1val, c2val in zip(c1, c2)]
            constraint_matrix.append(c)

        # Constrain the value of psi
        for r, z, psi in self.psivals:
            p1 = eq.psiRZ(r, z)
            constraint_rhs.append(psi - p1)

            # Coil responses
            c = tokamak.controlPsi(r, z)
            constraint_matrix.append(c)

        if not constraint_rhs:
            raise ValueError("No constraints given")

        # Constraint matrix
        A = array(constraint_matrix)
        b = np.reshape(array(constraint_rhs), (-1,))

        # Number of controls (length of x)
        ncontrols = A.shape[1]

        # First solve analytically by Tikhonov regularisation
        # minimise || Ax - b ||^2 + ||gamma x ||^2

        # Calculate the change in coil current
        self.current_change = dot(
            inv(dot(transpose(A), A) + self.gamma ** 2 * eye(ncontrols)),
            dot(transpose(A), b),
        )

        # Now use the initial analytical soln to guide constrained solve

        # Establish constraints on changes in coil currents from the present
        # and max/min coil current constraints

        current_change_bounds = []

        if self.current_lims is None:
            for i in range(ncontrols):
                current_change_bounds.append((-inf, inf))
        else:
            for i in range(ncontrols):
                cur = tokamak.controlCurrents()[i]
                lower_lim = self.current_lims[i][0] - cur
                upper_lim = self.current_lims[i][1] - cur
                current_change_bounds.append((lower_lim, upper_lim))

        current_change_bnds = array(current_change_bounds)

        # Reform the constraint matrices to include Tikhonov regularisation
        A2 = np.concatenate([A, self.gamma * eye(ncontrols)])
        b2 = np.concatenate([b, np.zeros(ncontrols)])

        # The objetive function to minimize
        # || A2x - b2 ||^2
        def objective(x):
            return (norm((A2 @ x) - b2)) ** 2

        # Additional constraints on the optimisation
        cons = []

        def max_total_currents(x):
            sum = 0.0
            for delta, i in zip(x, tokamak.controlCurrents()):
                sum += abs(delta + i)
            return -(sum - self.max_total_current)

        if self.max_total_current is not None:
            con1 = {"type": "ineq", "fun": max_total_currents}
            cons.append(con1)

        # Use the analytical current change as the initial guess
        if self.current_change.shape[0] > 0:
            x0 = self.current_change
            sol = optimize.minimize(
                objective, x0, method="SLSQP", bounds=current_change_bnds, constraints=cons
            )

            self.current_change = sol.x
            tokamak.controlAdjust(self.current_change)

        # Store info for user
        self.current_change = self.current_change

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def plot(self, axis=None, show=True):
        """
        Plots constraints used for coil current control

        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning

        """
        from .plotting import plotConstraints

        return plotConstraints(self, axis=axis, show=show)


class ConstrainPsi2D(object):
    """
    Adjusts coil currents to minimise the square differences
    between psi[R,Z] and a target psi.

    Ignores constant offset differences between psi array
    """

    def __init__(self, target_psi, weights=None):
        """
        target_psi : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psi
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        """
        if weights is None:
            weights = np.full(target_psi.shape, 1.0)

        # Remove the average so constant offsets are ignored
        self.target_psi = target_psi - np.average(target_psi, weights=weights)

        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psi_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psi_difference(self, currents, eq):
        """
        Difference between psi from equilibrium with the given currents
        and the target psi
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()
        psi_av = np.average(psi, weights=self.weights)
        return (
            (psi - psi_av - self.target_psi) * self.weights
        ).ravel()  # flatten array


class ConstrainPsiNorm2D(object):
    """
    Adjusts coil currents to minimise the square differences
    between normalised psi[R,Z] and a target normalised psi.
    """

    def __init__(self, target_psinorm, weights=1.0):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        """
        self.target_psinorm = target_psinorm
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psinorm_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psinorm_difference(self, currents, eq):
        """
        Difference between normalised psi from equilibrium with the given currents
        and the target psinorm
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()

        eq._updateBoundaryPsi(psi)
        psi_bndry = eq.psi_bndry
        psi_axis = eq.psi_axis

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        return (
            (psi_norm - self.target_psinorm) * self.weights
        ).ravel()  # flatten array


class ConstrainPsi2DAdvanced(object):
    """
    Adjusts coil currents to minimise the square differences
    between psi[R,Z] and a target psi.

    Attempts to also constrain the coil currents as in the 'constrain' class.
    """

    def __init__(
        self, target_psi, weights=1.0, current_lims=None, max_total_current=None
    ):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        current_bounds: List of tuples
            Optional list of tuples representing constraints on coil currents to be used
            when reconstructing the equilibrium from the geqdsk file.
            [(l1,u1),(l2,u2)...(lN,uN)]

        Create an instance, specifying the constraints to apply
        """

        self.current_lims = current_lims
        self.target_psi = target_psi
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = np.asarray(tokamak.controlCurrents())
        ncontrols = len(start_currents)

        # In order for the optimisation to work, the initial guess must be within the
        # bounds supplied. Hence, check start_currents and adjust accordingly to be within bounds
        for i in range(ncontrols):
            bnd_upper = max(self.current_lims[i])
            bnd_lower = min(self.current_lims[i])
            sc = start_currents[i]
            if not (bnd_lower <= sc <= bnd_upper):
                if sc < bnd_lower:
                    start_currents[i] = bnd_lower
                else:
                    start_currents[i] = bnd_upper

        current_bounds = []

        for i in range(ncontrols):
            if self.current_lims is None:
                current_bounds.append((-inf, inf))
            else:
                bnd_upper = max(self.current_lims[i])
                bnd_lower = min(self.current_lims[i])
                current_bounds.append((bnd_lower, bnd_upper))

        current_bnds = array(current_bounds)

        # Least squares optimisation of difference in target v achieved normalised psi
        # applied with bounds on coil currents
        end_currents = optimize.minimize(
            self.psi_difference,
            start_currents,
            method="L-BFGS-B",
            bounds=current_bnds,
            args=(eq,),
        ).x

        # Set the latest coil currents
        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psi_difference(self, currents, eq):
        """
        Sum of the squares of the differences between the achieved
        psi and the target psi.
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()
        eq._updateBoundaryPsi(psi)

        psi_av = np.average(psi, weights=self.weights)
        diff = (psi - psi_av - self.target_psi) * self.weights
        sum_square_diff = np.sum(diff * diff)

        return sum_square_diff


class ConstrainPsiNorm2DAdvanced(object):
    """
    Adjusts coil currents to minimise the square differences
    between normalised psi[R,Z] and a target normalised psi.

    Attempts to also constrain the coil currents as in the 'constrain' class.
    """

    def __init__(self, target_psinorm, weights=1.0, current_lims=None):
        """
        target_psinorm : 2D (R,Z) array
            Must be the same size as the equilibrium psi

        weights : float or 2D array of same size as target_psinorm
            Relative importance of each (R,Z) point in the fitting
            By default every point is equally weighted
            Set points to zero to ignore them in fitting.

        current_bounds: List of tuples
            Optional list of tuples representing constraints on coil currents to be used
            when reconstructing the equilibrium from the geqdsk file.
            [(l1,u1),(l2,u2)...(lN,uN)]

        Create an instance, specifying the constraints to apply
        """

        self.current_lims = current_lims
        self.target_psinorm = target_psinorm
        self.weights = weights

    def __call__(self, eq):
        """
        Apply constraints to Equilibrium eq
        """

        tokamak = eq.getMachine()
        start_currents = np.asarray(tokamak.controlCurrents())
        ncontrols = len(start_currents)

        # In order for the optimisation to work, the initial guess must be within the
        # bounds supplied. Hence, check start_currents and adjust accordingly to be within bounds
        for i in range(ncontrols):
            bnd_upper = max(self.current_lims[i])
            bnd_lower = min(self.current_lims[i])
            sc = start_currents[i]
            if not (bnd_lower <= sc <= bnd_upper):
                if sc < bnd_lower:
                    start_currents[i] = bnd_lower
                else:
                    start_currents[i] = bnd_upper

        current_bounds = []

        for i in range(ncontrols):
            if self.current_lims is None:
                current_bounds.append((-inf, inf))
            else:
                bnd_upper = max(self.current_lims[i])
                bnd_lower = min(self.current_lims[i])
                current_bounds.append((bnd_lower, bnd_upper))

        current_bnds = array(current_bounds)

        # Least squares optimisation of difference in target v achieved normalised psi
        # applied with bounds on coil currents
        end_currents = optimize.minimize(
            self.psinorm_difference,
            start_currents,
            method="L-BFGS-B",
            bounds=current_bnds,
            args=(eq,),
        ).x

        # Set the latest coil currents
        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psinorm_difference(self, currents, eq):
        """
        Sum of the squares of the differences between the achieved normalised
        psi and the target normalised psi.
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()

        eq._updateBoundaryPsi(psi)
        psi_bndry = eq.psi_bndry
        psi_axis = eq.psi_axis

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
        diff = (psi_norm - self.target_psinorm) * self.weights
        sum_square_diff = np.sum(diff * diff)

        return sum_square_diff

def fshape90(rshape=None,iway=2,ntet=256):
    '''
    Original fshape90 MATLAB by:
    V.V.Drozdov, A.A.Martynov, S.Yu.Medvedev
    Keldysh Institute of Applied Mathematics 2014

    fshape90 is a faithful Python translation of the original fshape90 MATLAB script.
    Translation by C.Marsden.
    
    rshape[0] -> rbc -> R0
    rshape[1] -> rl -> a
    rshape[2] -> dtre -> triangularity upper = tre + dtre
    rshape[3] -> tre -> triangularity lower = tre - dtre
    rshape[4] -> 0
    rshape[5] -> zbc -> Z0
    rshape[6] -> phase -> Rotates (R,Z) by phase
    rshape[7] -> vit -> elongation upper = vit + dvit
    rshape[8] -> dvit -> elongation lower = vit - divt
    rshape[9] -> 0
    rshape[10] -> fsepu -> 0 - smooth upper, >0 - 90 degree X-point
    rshape[11] -> fsepd -> 0 - smooth lower, >0 - 90 degree X-point

    iway -> default 2 paramterisation
    ntet -> Number of boundary points

    Fourier-like boundary paramterisation in the form:
    uksr = rshape[1]*(sepfac * cost - (rshape[3]+rshape[2]*sint)*sint^2)
    vksr = rshape[1]*sint*(rshape[7]+rshape[8]*sint)

    Returns r, z of the boundary points

    '''
    if rshape is None:

        rshape = [1.000, 0.750, -0.10, 0.40, 0.0, 0.000, 0.0, 2.00, -0.10, 0.0, 1.85, 1.45]

    tet = np.linspace(0.0,2*np.pi,ntet+1,endpoint=True)
    sint = np.sin(tet)
    cost = np.cos(tet)

    # For providing right angle
    d_up = (rshape[3]+1.5*rshape[2])**2. + (0.5*rshape[7]+rshape[8])**2.
    d_dn = (rshape[3]-1.5*rshape[2])**2. + (0.5*rshape[7]-rshape[8])**2.

    if iway==1:

        # Method 1

        fac1 = np.full(ntet+1,1.0)
        fac2 = np.full(ntet+1,1.0)

        if (rshape[10]>1 and rshape[11]>1):

            # Providing right angle for double null if possible

            a = 0.25 / d_up
            b = 0.25 / d_dn

            D = 1. - 2.*(a+b) + (a-b)**2.

            if D > 0:

                fsepu_inv = b - a + np.sqrt(D)
                fsepd_inv = a - b + np.sqrt(D)

                if (0<fsepu_inv<1 and 0<fsepd_inv<1):

                    rshape[10] = 1./fsepu_inv
                    rshape[11] = 1./fsepd_inv

                else:
                    print('')
                    # No right angle

            else:
                print('') 
                # No right angle

            fac1 = np.sqrt(rshape[10]*(1.-sint) / (rshape[10]-sint))
            fac2 = np.sqrt(rshape[11]*(1.+sint) / (rshape[11]+sint))

        elif(rshape[11]>1):

            # Providing right angle for single null

            if d_dn>0.5:

                rshape[11] = d_dn/(d_dn-0.5)

            else:

                rshape[11] = 1000.

            fac2 = np.sqrt(rshape[11]*(1.+sint) / (rshape[11]+sint))

        sepfac = fac1*fac2

    else:

        # Method 2

        fac1 = np.full(ntet+1,1.0)

        if (rshape[10]>0):

            # Providing right angle

            rshape[10] = np.sqrt(d_up)
            fac1 = rshape[10]*abs(cost) - (rshape[10]-1.)*cost*cost

        fac2 = np.full(ntet+1,1.0)

        if (rshape[11]>0):

            # Providing right angle

            rshape[11] = np.sqrt(d_dn)
            fac2 = rshape[11]*abs(cost) - (rshape[11]-1.)*cost*cost

        sepfac = fac2
        isge0 = [i for i,val in enumerate(sint) if val >= 0]
        for i in isge0:
            sepfac[i] = fac1[i]

    uksr = rshape[1]*(sepfac*cost-(rshape[3]+rshape[2]*sint)*sint*sint)
    vksr = rshape[1]*sint*(rshape[7]+rshape[8]*sint)

    cosrot = np.cos(np.pi*rshape[6])
    sinrot = np.sin(np.pi*rshape[6])

    r = rshape[0] + uksr*cosrot - vksr*sinrot
    z = rshape[5] + uksr*sinrot + vksr*cosrot

    return r,z

def make_lcfs(R0=4.21,
            A=1.80,
            Z0=0.0,
            delta_u=0.56,
            delta_l=0.56,
            kappa_u=3.00,
            kappa_l=3.00,
            fsep_u=1.45,
            fsep_l=1.45,
            phase=0.0,
            Npoints=256,
            method=2,
            show=False
):

    '''
    Wrapper function for fshape90. Designed to accept
    the more commonly used plasma shape paramters.

    R0 -> Major radius [m]
    A -> Aspect ratio [-]
    Z0 -> Midplane height [m]
    delta_u -> Upper triangularity [-]
    delta_l -> Lower triangularity [-]
    kappa_u -> Upper elongation [-]
    kappa_l -> Lower elongation [-]
    fsep_u -> Upper smoothing factor. 0 if no X-point, >0 if X-point [-]
    fsep_l -> Lower smoothing factor. 0 if no X-point, >0 if X-point [-]
    phase -> Phase to rotate boundary points by, fraction of 180 deg.
    Npoints -> Number of boundary points [-]
    method -> 1, 2 , method to be used. Default is 2 [-]
    show -> Plot LCFS [bool]

    Returns r, z of the boundary points
    '''

    # Define minor radius
    a = R0 / A
    
    # Define intermediate variables used as inputs to fshape90
    tre = 0.5*(delta_u + delta_l)
    dtre = 0.5*(delta_u - delta_l)
    vit = 0.5*(kappa_u + kappa_l)
    dvit = 0.5*(kappa_u - kappa_l)

    # Define inputs to fshape90
    rshape = [R0,a,dtre,tre,0,Z0,phase,vit,dvit,0,fsep_u,fsep_l]

    # Generate LCFS with fshape90
    r,z = fshape90(rshape=rshape,iway=method,ntet=Npoints)

    if show:

        # Plot the LCFS generated by fshape90
        title_str = r'$R_{0}$: '+str(R0)+'m. '
        title_str += r'A: '+str(A)+'. '
        title_str += r'$Z_{0}$: '+str(Z0)+'m. '
        title_str += r'$\delta_{u}$: '+str(delta_u)+'. '
        title_str += r'$\delta_{l}$: '+str(delta_l)+'. '
        title_str += '\n'
        title_str += r'$\kappa_{u}$: '+str(kappa_u)+'. '
        title_str += r'$\kappa_{l}$: '+str(kappa_l)+'. '
        title_str += r'Phase: '+str(phase*180.0)+'deg. '
        title_str += r'$f_{sep,u}$: '+str(fsep_u)+'. '
        title_str += r'$f_{sep,l}$: '+str(fsep_l)+'. '

        fig, ax = plt.subplots()
        ax.plot(r,z,'b')
        ax.plot(r,z,'bx')
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal')
        ax.set_title(title_str)
        plt.show()

    return r,z

def generate_separatrix_points(R0=4.21,
            A=1.80,
            Z0=0.0,
            delta_u=0.56,
            delta_l=0.56,
            kappa_u=3.00,
            kappa_l=3.00,
            fsep_u=1.45,
            fsep_l=1.45,
            phase=0.0,
            Npoints=256
):
    '''
    Inputs:
    Takes as input plasma shaping parameters that described the shape of the separatrix.

    Outputs:
    Returns a dictionary containing:
    - Rlcfs -> List of R coordinates of points on the LCFS
    - Zlcfs -> List of Z coordinates of points on the LCFS
    - IMP -> (R,Z) tuple of the HFS midplane
    - OMP -> (R,Z) tuple of the LFS midplane
    - UPPER -> (R,Z) tuple of the point on the LCFS of the most positive Z
    - LOWER -> (R,Z) tuple of the point on the LCFS of the most negative Z
    '''

    # Generate r,z points of the LCFS
    r,z = make_lcfs(R0,
            A,
            Z0,
            delta_u,
            delta_l,
            kappa_u,
            kappa_l,
            fsep_u,
            fsep_l,
            phase,
            Npoints,
            method=2,
            show=False)
    
    # Extract the IMP, OMP, UPPER and LOWER points
    arg_IMP = np.argmin(r)
    arg_OMP = np.argmax(r)
    arg_UPPER = np.argmax(z)
    arg_LOWER = np.argmin(z)

    R_IMP = r[arg_IMP]
    Z_IMP = z[arg_IMP]

    R_OMP = r[arg_OMP]
    Z_OMP = z[arg_OMP]

    R_UPPER = r[arg_UPPER]
    Z_UPPER = z[arg_UPPER]

    R_LOWER = r[arg_LOWER]
    Z_LOWER = z[arg_LOWER]

    LCFS = {'Rlcfs':r,
            'Zlcfs':z,
            'IMP':(R_IMP,Z_IMP),
            'OMP':(R_OMP,Z_OMP),
            'UPPER':(R_UPPER,Z_UPPER),
            'LOWER':(R_LOWER,Z_LOWER)
            }
    
    return LCFS