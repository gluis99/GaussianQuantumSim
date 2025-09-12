import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Tuple, Optional
from numpy.typing import NDArray

class CatState:
    """
    Quantum optical cat state (superposition of coherent states).

    An n-component cat state:
        - Even: |cat_even> = (1/N) sum_k |alpha * exp(2pi i k / n)>
        - Odd:  |cat_odd>  = (1/N) sum_k exp(2pi i k / n) |alpha * exp(2pi i k / n)>

    Attributes
    ----------
    alpha : complex
        Base coherent amplitude.
    n : int
        Number of coherent states in superposition.
    even : bool
        True for even cat, False for odd cat.
    coherents : NDArray[np.complex128]
        Coherent amplitudes in superposition.
    coeffs : NDArray[np.complex128]
        Complex coefficients for superposition.
    """

    def __init__(self, alpha: complex, n: int = 2, even: bool = True) -> None:
        """
        Initialize a cat state.

        Parameters
        ----------
        alpha : complex
            Coherent amplitude of the base state.
        n : int, optional
            Number of coherent states in the superposition (default is 2).
        even : bool, optional
            If True, constructs an even cat state. If False, constructs an odd cat state.
        """
        self.alpha = alpha
        self.n = n
        self.even = even

        phases = np.array([2*np.pi*k/n for k in range(n)])
        self.coherents: NDArray[np.complex128] = np.array([alpha * np.exp(1j*ph) for ph in phases])
        self.coeffs: NDArray[np.complex128] = np.ones(n, dtype=complex) if even else np.exp(1j*phases)

        # Precompute overlap exponent matrix
        self._overlaps: NDArray[np.complex128] = np.exp(
            -0.5*(np.abs(self.coherents[:, None])**2 + np.abs(self.coherents[None, :])**2)
            - np.conjugate(self.coherents[None, :]) * self.coherents[:, None]
        )
        # Coefficient matrix
        self._coeff_mat: NDArray[np.complex128] = self.coeffs[:, None] * np.conjugate(self.coeffs)[None, :]

    @property
    def norm2(self) -> float:
        """
        Squared normalization factor of the cat state.

        Returns
        -------
        float
            Normalization factor squared.
        """
        return float(np.real(np.einsum('ij,ij->', self._overlaps * self._coeff_mat, np.ones_like(self._overlaps))))

    def wigner_function(self, xi: Union[complex, NDArray]) -> Union[float, NDArray]:
        """
        Evaluate the Wigner function of the cat state at given phase-space points.

        Parameters
        ----------
        xi : complex or NDArray
            Phase-space points. Can be:
            - Scalar complex number (single point)
            - 1D or 2D array of complex numbers (vectorized)
            - Nx2 array of real numbers representing (x, p) coordinates

        Returns
        -------
        W : float or NDArray
            Wigner function value(s) at the specified point(s). Shape matches input.
            If input is scalar, returns a float.

        Notes
        -----
        The Wigner function is computed as:
            W(ξ) = (2 / (π * ||ψ||^2)) Σ_{k,l} c_k c_l* ⟨α_l| D(2ξ) |α_k⟩
        where D(ξ) is the displacement operator and ||ψ||^2 is the normalization.

        Examples
        --------
        >>> cat = CatState(1.0, n=2, even=True)
        >>> W0 = cat.wigner_function(0+0j)
        >>> xi_grid = np.linspace(-2,2,10) + 1j*np.linspace(-2,2,10)[:,None]
        >>> W_grid = cat.wigner_function(xi_grid)
        """
        xi_arr = np.asarray(xi)
        if xi_arr.ndim >= 1 and xi_arr.shape[-1] == 2:
            xi_flat = (xi_arr[..., 0] + 1j*xi_arr[..., 1]).ravel()
            out_shape = xi_arr.shape[:-1]
        else:
            xi_flat = xi_arr.ravel().astype(complex)
            out_shape = xi_arr.shape

        xi_b = xi_flat[:, None, None]
        ak_b = self.coherents[None, :, None]
        al_b = self.coherents[None, None, :]

        exponent = (
            -2.0 * np.abs(xi_b)**2
            + 2.0 * np.conjugate(xi_b) * ak_b
            + 2.0 * xi_b * np.conjugate(al_b)
        ) + self._overlaps  # includes -0.5(|ak|^2 + |al|^2) - conj(al)*ak

        term = np.exp(exponent)
        W_flat = np.tensordot(term, self._coeff_mat, axes=([1, 2], [0, 1]))
        W_flat = (2.0 / (np.pi * self.norm2)) * W_flat
        W_flat = np.real_if_close(W_flat, tol=1e6)
        W = W_flat.reshape(out_shape)
        return float(W) if W.shape == () else W

    def wigner_min(self, rng: Tuple[float, float] = (-5, 5), num_points: int = 200) -> float:
        """
        Compute the minimum value of the Wigner function over a 2D grid.

        Parameters
        ----------
        rng : tuple of float
            (min, max) range for x and p axes.
        num_points : int
            Number of points per axis.

        Returns
        -------
        float
            Minimum Wigner function value on the grid.
        """
        s = np.linspace(*rng, num_points)
        X, P = np.meshgrid(s, s)
        xi = X + 1j*P
        W = self.wigner_function(xi).reshape((num_points, num_points))
        return float(np.min(W))

    def plot_wigner_2d(self, rng: Tuple[float, float] = (-5, 5), num_points: int = 200,
                        W: Optional[NDArray] = None, show: bool = True) -> None:
        """
        Plot the 2D Wigner function in phase space.

        Parameters
        ----------
        rng : tuple of float
            Range (min, max) for both x and p axes.
        num_points : int
            Number of points per axis.
        W : NDArray, optional
            Precomputed Wigner function to plot (saves recomputation).
        show : bool
            If True, calls plt.show().

        Returns
        -------
        None
        """
        s = np.linspace(*rng, num_points)
        X, P = np.meshgrid(s, s)
        if W is None:
            xi = X + 1j*P
            W = self.wigner_function(xi).reshape((num_points, num_points))
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(X, P, W.real, shading='auto', cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.title('Wigner Function')
        plt.colorbar(label='Wigner Value')
        plt.tight_layout()
        if show:
            plt.show()

    def plot_wigner_3d(self, rng: Tuple[float, float] = (-5, 5), num_points: int = 200,
                        W: Optional[NDArray] = None, show: bool = True) -> None:
        """
        Plot the Wigner function as a 3D surface in phase space.

        Parameters
        ----------
        rng : tuple of float
            Range (min, max) for both x and p axes.
        num_points : int
            Number of points per axis.
        W : NDArray, optional
            Precomputed Wigner function to plot (saves recomputation).
        show : bool
            If True, calls plt.show().

        Returns
        -------
        None
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        s = np.linspace(*rng, num_points)
        X, P = np.meshgrid(s, s)
        if W is None:
            xi = X + 1j*P
            W = self.wigner_function(xi).reshape((num_points, num_points))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, P, W.real, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('p')
        ax.set_zlabel('Wigner Value')
        ax.set_title(f'CatState Wigner (n={self.n})')
        plt.tight_layout()
        if show:
            plt.show()

