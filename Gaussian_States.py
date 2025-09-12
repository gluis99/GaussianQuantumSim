import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Tuple, Optional
from numpy.typing import NDArray

class GaussianState:
    """
    N-mode Gaussian quantum state.

    Attributes
    ----------
    V : NDArray[np.float64]
        Covariance matrix (2N x 2N).
    r : NDArray[np.float64]
        Displacement vector (2N).
    N : int
        Number of modes.
    """

    def __init__(self, V: NDArray, r: Optional[NDArray] = None) -> None:
        """
        Initialize a Gaussian state.

        Parameters
        ----------
        V : NDArray
            Covariance matrix (2N x 2N).
        r : NDArray, optional
            Displacement vector (2N). Defaults to zero.
        """
        self.V = np.array(V, dtype=float)
        self.N = self.V.shape[0] // 2
        self.r = np.zeros(2*self.N) if r is None else np.array(r, dtype=float)

    @classmethod
    def vacuum(cls, N: int) -> "GaussianState":
        """
        Create an N-mode vacuum state.

        Parameters
        ----------
        N : int
            Number of modes.

        Returns
        -------
        GaussianState
            N-mode vacuum state with identity covariance and zero displacement.
        """
        V = np.identity(2*N)
        r = np.zeros(2*N)
        return cls(V, r)

    def wigner_function(self, xi: NDArray) -> Union[float, NDArray]:
        """
        Evaluate the Wigner function at given phase-space points.

        Parameters
        ----------
        xi : NDArray, shape (2N,) or (M, 2N)
            Phase-space points.

        Returns
        -------
        W : float or NDArray
            Wigner function value(s). Shape matches input if vectorized.
        """
        xi_arr = np.atleast_2d(xi)
        xi_disp = xi_arr - self.r
        Vinv_xi = np.linalg.solve(self.V, xi_disp.T).T
        exponent = np.einsum("ij,ij->i", xi_disp, Vinv_xi)
        sign, logdet = np.linalg.slogdet(self.V)
        norm = 1 / ((2*np.pi)**self.N * np.sqrt(np.exp(logdet)))
        W = norm * np.exp(-0.5 * exponent)
        W = W if len(W) > 1 else W[0]
        return W

    def apply_gaussian_gate(self, S: NDArray, d: Optional[NDArray] = None) -> "GaussianState":
        """
        Apply a linear (symplectic) Gaussian gate in-place.

        Parameters
        ----------
        S : NDArray, shape (2N,2N)
            Symplectic matrix.
        d : NDArray, shape (2N,), optional
            Displacement vector. Defaults to zero.

        Returns
        -------
        GaussianState
            Self after transformation.
        """
        d = np.zeros_like(self.r) if d is None else d
        self.V = S @ self.V @ S.T
        self.r = S @ self.r + d
        return self

    def transformed_state(self, S: NDArray, d: Optional[NDArray] = None) -> "GaussianState":
        """
        Return a new GaussianState after applying the transformation (S, d).

        Parameters
        ----------
        S : NDArray, shape (2N,2N)
            Symplectic matrix.
        d : NDArray, shape (2N,), optional
            Displacement vector. Defaults to zero.

        Returns
        -------
        GaussianState
            Transformed state.
        """
        d = np.zeros_like(self.r) if d is None else d
        V_new = S @ self.V @ S.T
        r_new = S @ self.r + d
        return GaussianState(V_new, r_new)

    def plot_wigner_2d(self, quad1: int = 0, quad2: int = 1,
                        x_range: Tuple[float, float] = (-5, 5), num_points: int = 100,
                        fixed_point: Optional[NDArray] = None) -> None:
        """
        Plot 2D Wigner function along two quadratures.

        Parameters
        ----------
        quad1, quad2 : int
            Indices of the quadratures to plot.
        x_range : tuple
            Range (min, max) for axes.
        num_points : int
            Number of points per axis.
        fixed_point : NDArray, optional
            Fixed coordinates for other quadratures.

        Returns
        -------
        None
        """
        N = self.N
        twoN = 2*N
        fixed_point = np.zeros(twoN) if fixed_point is None else fixed_point
        x = np.linspace(fixed_point[quad1]+x_range[0], fixed_point[quad1]+x_range[1], num_points)
        y = np.linspace(fixed_point[quad2]+x_range[0], fixed_point[quad2]+x_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        xi = np.zeros((num_points**2, twoN))
        xi[:, quad1] = X.ravel()
        xi[:, quad2] = Y.ravel()
        for q in range(twoN):
            if q not in (quad1, quad2):
                xi[:, q] = fixed_point[q]
        W = self.wigner_function(xi).reshape((num_points, num_points))
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(X, Y, W, shading='auto', cmap='viridis')
        labels = [f'x{i//2+1}' if i % 2 == 0 else f'p{i//2+1}' for i in range(twoN)]
        plt.xlabel(labels[quad1])
        plt.ylabel(labels[quad2])
        plt.title('Gaussian Wigner Function')
        plt.colorbar(label='Wigner Value')
        plt.tight_layout()
        plt.show()


