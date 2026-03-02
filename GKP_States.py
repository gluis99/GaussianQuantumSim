import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Gaussian_Gates_Symplectic import *
from States import Omega


class GKPState:
    """
    A class representing a GKP (Gottesman-Kitaev-Preskill) quantum state (grid state).
    As a grid defined by a generator matrix M and displacement vector.

    Attributes
    ----------
    l : float
        Lattice scale (for computational state l = 2 *sqrt(pi))

    M : NDArray
        Generator matrix defining the GKP state. 
        (Now 2x2 for N=1, may be extended to 2Nx2N for N>1 in the future)

    displacement : NDArray
        Physical displacement vector for the GKP state.

    displacement_norm : NDArray
        Normalized displacement vector, such that displacement = l * displacement_norm.

    To be implemented in the future:
    N : int
        Number of modes. For the moment, only implemented for N=1. 
        (may be extended to N>1 in the future)
    
    """
    def __init__(self, Mat: np.ndarray, displacement: np.ndarray, l = None, norm_disp: bool = False) -> None:
        """
        Initialize a GKP state.

        Parameters
        ----------
        Max : NDArray
            Unnormalized (unless l given) generator matrix defining the GKP state.
        displacement : NDArray
            Displacement vector for the GKP state.
        norm_disp : bool, optional
            If True, `displacement` is interpreted as normalized coordinates.
            If False, `displacement` is interpreted as physical coordinates and
            converted via `displacement_norm = displacement / l`.
        """
        Omega = epsilon # Symplectic form for N=1

        if l == None:
            # Lattice scaling
            l = np.abs(np.linalg.det(Mat))
            self.M = Mat /l
        else:
            # Need to implement way to assure that M is normalized
            self.M = Mat
        self.l = l
        if self.l == 0:
            raise ValueError("l must be non-zero.")
        displacement = np.asarray(displacement, dtype=float)
        if norm_disp:
            self.displacement_norm = displacement
        else:
            self.displacement_norm = displacement / self.l
        self.A = np.einsum('ij,jk,kl->il', self.M.T, Omega, self.M)  # Gram matrix, has to have integer entries

    @property
    def displacement(self) -> np.ndarray:
        """
        Physical displacement vector (same units as x and p).
        """
        return self.l * self.displacement_norm

    @displacement.setter
    def displacement(self, value: np.ndarray) -> None:
        """
        Set displacement from physical coordinates.
        """
        value = np.asarray(value, dtype=float)
        self.displacement_norm = value / self.l

    @classmethod
    def canonical_GKP(cls) -> "GKPState":
        """
        Create a canonical GKP state.

        Returns
        -------
        GKPState
            Canonical GKP state with generator matrix [[1, 0], [0, 1]].
        """
        M = np.array([[0, 1], [1, 0]])
        l = np.sqrt(2*np.pi)
        displacement = np.zeros(2)
        return cls(M, displacement, l)
    
    @classmethod
    def computational_GKP(cls, b, l = 2 *np.sqrt(np.pi)) -> "GKPState":
        """
        Create a computational GKP state.

        Parameters
        ----------
        b : int
            Binary value (0 or 1) for the computational GKP state.

        Returns
        -------
        GKPState
            Computational GKP qubit state |b>_L.
        """
        M = np.array([[0, 1], [1, 0]])
        if b == 0:
            displacement = np.zeros(2)
            norm_disp = True
        elif b == 1:
            # Displaced by action of X_L = sqrt(S_2)
            displacement = np.array([0.5, 0.0])
            norm_disp = True
        else:
            raise ValueError("b must be 0 or 1.")
        return cls(M, displacement, l, norm_disp=norm_disp)
    
    def gram_matrix(self) -> np.ndarray:
        """
        Compute the Gram matrix of the GKP state.

        Returns
        -------
        NDArray
            Gram matrix A = M^T @ Omega @ M, which should have integer entries for a valid GKP state.
        """
        #if self.N == 1:
        if True:
            Omega = epsilon
        else:
            Omega = Omega(self.N)
        A = np.einsum('ij,jk,kl->il', self.M.T, Omega, self.M)
        return A
    
    def dimension(self) -> int:
        """
        Compute the dimension of the logical subspace encoded by the GKP state.

        Returns
        -------
        int
            Dimension of the logical subspace, given by sqrt(det(A)) where A is the Gram matrix.
        """
        A = self.gram_matrix()

        scale_factor = self.l**2/(2*np.pi)
        return int(np.round(np.sqrt(np.linalg.det(A)) * scale_factor))
    
    def stabilizers(self):
        """
        Compute the stabilizers of the GKP state.

        Returns
        -------
        tuple of callables
            Stabilizer functions S_1 and S_2 corresponding to the grid vectors defined by M.
        """
        stabilizers = []
        for v in self.M.T:
            stabilizers.append(lambda x,p: np.exp(1j *self.l * simplectic_form(np.array([x,p]), v)))
        return tuple(stabilizers)

    # Only if states equivalent to computational GKP states
    def logical_operators(self):
        """
        Compute the logical operators of the GKP state.

        Returns
        -------
        tuple of callables
            Logical operator functions X_L and Z_L corresponding to the grid vectors defined by M.
        """
        logical_ops = []
        for v in self.M.T:
            logical_ops.append(lambda x,p: np.exp(0.5j *self.l * simplectic_form(np.array([x,p]), v)))
        return tuple(logical_ops)
    
    def apply_gaussian_gate(self, S: np.ndarray, d = np.array([0,0])) -> "GKPState":
        """
        Apply a linear (symplectic) Gaussian gate in-place.

        Parameters
        ----------
        S : NDArray or callable
            Symplectic matrix (2N x 2N) or a callable that builds it from N.
        d : NDArray, shape (2N,), optional
            Displacement vector. Defaults to zero.

        Returns
        -------
        GKPState
            Self after transformation.
        """
        #if self.N == 1:
        if True:
            Sinv = - np.einsum('ij,jk,kl->il', epsilon, S.T, epsilon)
        else:
            Sinv = - np.einsum('ij,jk,kl->il', Omega(self.N), S.T, Omega(self.N))

        self.M = np.einsum('ij,jk->ik', Sinv, self.M)
        self.displacement = np.einsum('ij,j->i', Sinv, self.displacement) + d

        return self

    def expectation_value(
        self,
        O,
        x_range=(-5, 5),
        p_range=(-5, 5),
        num_points=801,
        delta_x=0.2,
        delta_p=None,
    ):
        """
        Expectation value of observable O(x,p)
        with respect to this GKP state's Wigner function.
        """

        X, P, W = self.wigner_approx(
            x_range=x_range,
            p_range=p_range,
            num_points=num_points,
            delta_x=delta_x,
            delta_p=delta_p,
            normalize=True,
        )

        O_values = O(X, P)

        dx = X[0, 1] - X[0, 0]
        dp = P[1, 0] - P[0, 0]

        return np.sum(W * O_values) * dx * dp

    # Only for N=1 for the moment
    def Q_operator(self, print_Q = False):
        """
        Compute the Q operator for the GKP state at given x and p.

        Parameters
        ----------
        x : float
            Position variable.
        p : float
            Momentum variable.

        Returns
        -------
        float
            Value of the Q operator at (x, p).
        """
        dim = self.dimension()
        v_1, v_2 = self.M.T
        disp = self.displacement
        l = self.l

        if dim == 2:
            if print_Q:
                print(f'2 - cos(1/2 *{l}*[x*({v_1[1]}) - p*({v_1[0]}) +phi_1]) - cos(   [x*y({v_1[1]}) - p*({v_1[0]})] + phi_2)')
                print(f'phi1 = {simplectic_form(disp, v_1)}')
                print(f'phi2 = {simplectic_form(disp, v_2)}')
            return lambda x,p: 2 - np.cos(0.5*l *(simplectic_form(np.array([x,p]), v_1) + simplectic_form(disp, v_1) )) \
                                 - np.cos(    l * simplectic_form(np.array([x,p]), v_2) + simplectic_form(disp, v_2)  )
        elif dim == 1:
            return lambda x,p: 2 - np.cos(l*(simplectic_form(np.array([x,p]), v_1) + simplectic_form(disp, v_1) )) \
                                 - np.cos(l*(simplectic_form(np.array([x,p]), v_2) + simplectic_form(disp, v_2) ))
        else:
            raise NotImplementedError("Q operator only implemented for dimension 1 and 2.")

    # Function to compute an approximation of the expectation value of the Q operator
    # xi = integral dx dp W(x, p) Q(x, p)
    def xi_approx(self, wigner_plot, clip_to_physical=False):
        X, P, W = wigner_plot
        
        dx = X[0, 1] - X[0, 0]
        dp = P[1, 0] - P[0, 0]

        Q_values = self.Q_operator()(X, P)

        xi_raw = float(np.real_if_close(np.sum(W * Q_values) * dx * dp, tol=1e6))
        if clip_to_physical:
            return max(xi_raw, 0.0)
        return xi_raw


    ##################################################################################
    # Wigner function related, taken from AI-generated code, to be adapted

    def _resolve_finite_energy_widths(self, delta_x, delta_p):
        """
        Resolve finite-energy widths for approximate GKP states.
        """
        if delta_x is None and delta_p is None:
            delta_x = 0.2
            delta_p = 0.2
        elif delta_x is None:
            delta_x = delta_p
        elif delta_p is None:
            delta_p = delta_x

        if delta_x <= 0 or delta_p <= 0:
            raise ValueError("delta_x and delta_p must be positive.")

        return float(delta_x), float(delta_p)
    
    # Wigner function of approximate GKP state
    def wigner_finite_energy(
        self,
        x_range=(-5, 5),
        p_range=(-5, 5),
        num_points=801,
        delta_x=0.2,
        delta_p=None,
        envelope_cutoff=4.0,
        peak_cutoff=4.0,
        normalize=True,
    ):
        """
        Approximate finite-energy GKP Wigner function.

        Uses Gaussian peaks with widths `delta_x` and `delta_p` in x and p,
        and an inverse-width envelope consistent with finite-energy modeling:
        envelope ~ exp(-0.5 * ((delta_p*x0)^2 + (delta_x*p0)^2)).

        Lattice points are generated using an `l * M` Bravais lattice plus a
        four-point half-cell basis:

            point = n @ (l*M)^T + (r/2) @ (l*M)^T + displacement,

        with r in {0,1}^2 and basis sign (-1)^(r1*r2). This preserves
        stabilizer translations by l while retaining nontrivial logical
        half-translations.

        """

        delta_x, delta_p = self._resolve_finite_energy_widths(
            delta_x=delta_x,
            delta_p=delta_p,
        )

        x = np.linspace(x_range[0], x_range[1], num_points)
        p = np.linspace(p_range[0], p_range[1], num_points)
        X, P = np.meshgrid(x, p)

        dx = x[1] - x[0]
        dp = p[1] - p[0]

        W = np.zeros_like(X)

        # Determine which lattice points could contribute

        # Bounding rectangle from envelope (inverse-width finite-energy envelope)
        envelope_x_bound = envelope_cutoff / delta_p
        envelope_p_bound = envelope_cutoff / delta_x

        # Bounding rectangle from plotting window + peak widths
        window_x_bound = max(abs(x_range[0]), abs(x_range[1])) + peak_cutoff * delta_x
        window_p_bound = max(abs(p_range[0]), abs(p_range[1])) + peak_cutoff * delta_p

        bound_x = max(envelope_x_bound, window_x_bound)
        bound_p = max(envelope_p_bound, window_p_bound)

        effective_M = self.l * self.M

        # Half-cell offsets can extend points from Bravais-node centers.
        cell_half_span = 0.5 * np.max(np.linalg.norm(effective_M.T, axis=1))
        max_radius = np.sqrt(bound_x**2 + bound_p**2) + cell_half_span

        # Convert radius to lattice index radius

        singular_values = np.linalg.svd(effective_M, compute_uv=False)
        s_min = max(np.min(singular_values), 1e-12)

        lattice_radius = int(np.ceil(max_radius / s_min))

        k_vals = np.arange(-lattice_radius, lattice_radius + 1)
        basis_indices = ((0, 0), (1, 0), (0, 1), (1, 1))

        for k1 in k_vals:
            for k2 in k_vals:

                n = np.array([k1, k2], dtype=float)

                for r1, r2 in basis_indices:
                    r = np.array([r1, r2], dtype=float)
                    point = n @ effective_M.T + 0.5 * (r @ effective_M.T) + self.displacement

                    # Envelope cutoff pruning
                    envelope_argument = (delta_p * point[0])**2 + (delta_x * point[1])**2
                    if envelope_argument > envelope_cutoff**2:
                        continue

                    # Window pruning
                    if (
                        point[0] < x_range[0] - peak_cutoff * delta_x
                        or point[0] > x_range[1] + peak_cutoff * delta_x
                        or point[1] < p_range[0] - peak_cutoff * delta_p
                        or point[1] > p_range[1] + peak_cutoff * delta_p
                    ):
                        continue

                    parity = (-1.0) ** (r1 * r2)
                    envelope = np.exp(-0.5 * envelope_argument)

                    W += (
                        parity
                        * envelope
                        * np.exp(
                            -0.5 * (
                                ((X - point[0]) / delta_x)**2
                                + ((P - point[1]) / delta_p)**2
                            )
                        )
                    )

        W *= 1.0 / (2 * np.pi * delta_x * delta_p)

        if normalize:
            norm = np.sum(W) * dx * dp
            if norm != 0:
                W /= norm

        return X, P, W

    def plot_wigner_finite_energy(
        self,
        x_range=(-5, 5),
        p_range=(-5, 5),
        num_points=801,
        delta_x=0.2,
        delta_p=None,
        scale_axes_by_l=True,
    ):
        """
        Plot the canonical finite-energy GKP Wigner function
        using imshow (fast and clean).

        Parameters
        ----------
        x_range, p_range : tuple
            Plot window.
        num_points : int
            Grid resolution per axis.
        delta_x, delta_p : float, optional
            Peak widths in x and p. If only one is given, symmetric widths are used.
        scale_axes_by_l : bool
            If True, axes shown in units of l.
        """

        # Get normalized canonical Wigner function
        X, P, W = self.wigner_finite_energy(
            x_range=x_range,
            p_range=p_range,
            num_points=num_points,
            delta_x=delta_x,
            delta_p=delta_p,
            normalize=True,
        )

        # Symmetric color scale
        vmax = np.max(np.abs(W))
        if vmax == 0:
            vmax = 1.0

        # Axis scaling
        if scale_axes_by_l:
            unit = self.l
            extent = [
                x_range[0] / unit,
                x_range[1] / unit,
                p_range[0] / unit,
                p_range[1] / unit,
            ]
            xlabel = r"$x/\ell$"
            ylabel = r"$p/\ell$"
        else:
            extent = [
                x_range[0],
                x_range[1],
                p_range[0],
                p_range[1],
            ]
            xlabel = r"$x$"
            ylabel = r"$p$"

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

        im = ax.imshow(
            W,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Wigner function")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Finite-energy GKP Wigner function")

        plt.tight_layout()
        plt.show()

