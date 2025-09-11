import numpy as np
import matplotlib.pyplot as plt

class GaussianState:
    def __init__(self, V, r=None):
        """
        Initialize a Gaussian state.

        Parameters
        ----------
        V : ndarray, shape (2N, 2N)
            Covariance matrix of the state.
        r : ndarray, shape (2N,), optional
            Displacement vector. If None, defaults to the zero vector.
        """
        self.V = np.array(V, dtype=float)
        self.r = np.zeros(V.shape[0]) if r is None else np.array(r, dtype=float)
        self.N = len(self.r) // 2  # Number of modes
    
    def __repr__(self):
        return f"GaussianState(N={self.N}, r={self.r}, V=<{self.V.shape} matrix>)"

    @classmethod
    def vacuum(cls, N):
        """
        Create an N-mode vacuum state.

        Parameters
        ----------
        N : int
            Number of modes. Must be positive.

        Returns
        -------
        GaussianState
            The N-mode vacuum state with identity covariance and zero displacement.
        """
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer")
        V = np.identity(2 * N)
        r = np.zeros(2 * N)
        return cls(V, r)

    def wigner_function(self, xi):
        """
        Evaluate the Wigner function of the Gaussian state.

        Parameters
        ----------
        xi : ndarray, shape (2N,) or (M, 2N)
            Phase-space point(s) where the Wigner function is evaluated.

        Returns
        -------
        W : float or ndarray, shape (M,)
            Wigner function value(s) at the given point(s).
        """
        xi = np.atleast_2d(xi)
        xi_disp = xi - self.r
        Vinv_xi = np.linalg.solve(self.V, xi_disp.T).T
        exponent = np.einsum("ij,ij->i", xi_disp, Vinv_xi)

        sign, logdet = np.linalg.slogdet(self.V)
        norm = 1 / ((2*np.pi)**self.N * np.sqrt(np.exp(logdet)))

        W = norm * np.exp(-0.5 * exponent)
        return W if len(W) > 1 else W[0]

    def apply_gaussian_gate(self, S, d=None):
        """
        Transform the state by applying an affine symplectic transformation
          defined by (S, d): V -> S V S^T, r -> S r + d

        Parameters
        ----------
        S : ndarray, shape (2N, 2N)
            Symplectic matrix representing the linear part of the transformation.
        d : ndarray, shape (2N,), optional
            Displacement vector representing the affine part of the transformation.
            Defaults to the zero vector.

        Returns
        -------
        GaussianState
            The transformed Gaussian state (in-place).
        """
        if d is None:
            d = np.zeros_like(self.r)

        self.V = S @ self.V @ S.T
        self.r = S @ self.r + d
        return self
    
    def transformed_state(self, S, d=None):
        """
        Return a new GaussianState after applying the transformation (S, d),
        leaving the original state unchanged.
        Parameters
        ----------
        S : ndarray, shape (2N, 2N)
            Symplectic matrix representing the linear part of the transformation.
        d : ndarray, shape (2N,), optional
            Displacement vector representing the affine part of the transformation.
            Defaults to the zero vector.
        Returns
        -------
        GaussianState
            The new transformed Gaussian state.
        """
        if d is None:
            d = np.zeros_like(self.r)
        V_new = S @ self.V @ S.T
        r_new = S @ self.r + d
        return GaussianState(V_new, r_new)

    def plot_wigner_2d(self, quad1=0, quad2=1, x_range=(-5, 5), num_points=100, fixed_point=None):
        """
        Plot the Wigner function in the plane of two quadratures.

        Parameters
        ----------
        quad1, quad2 : int
            Indices of the quadratures to plot (e.g., 0=x1, 1=p1, 2=x2, ...).
        x_range : tuple of float
            Range (min, max) for both quadratures.
        num_points : int
            Number of grid points per axis.
        fixed_point : array-like, shape (2N,), optional
            Fixed point in phase space. The two chosen quadratures are swept
            over `x_range`, while the others remain fixed. Defaults to zeros.

        Returns
        -------
        None
            Displays a 2D color plot of the Wigner function.
        """
        N = self.N
        twoN = 2 * N
        if fixed_point is None:
            fixed_point = np.zeros(twoN)
        else:
            fixed_point = np.array(fixed_point)
            if len(fixed_point) != twoN:
                raise ValueError("fixed_point must have length 2*N")

        x = np.linspace(fixed_point[quad1]+x_range[0], fixed_point[quad1]+x_range[1], num_points)
        y = np.linspace(fixed_point[quad2]+x_range[0], fixed_point[quad2]+x_range[1], num_points)
        X, Y = np.meshgrid(x, y)

        xi = np.zeros((num_points**2, twoN))
        xi[:, quad1] = X.ravel()
        xi[:, quad2] = Y.ravel()
        for q in range(twoN):
            if q not in (quad1, quad2):
                xi[:, q] = fixed_point[q]

        W = self.wigner_function(xi)
        W = W.reshape((num_points, num_points))

        plt.figure(figsize=(6, 5))
        plt.pcolormesh(X, Y, W, shading='auto', cmap='viridis')
        labels = [f'x{i//2+1}' if i % 2 == 0 else f'p{i//2+1}' for i in range(twoN)]
        plt.xlabel(labels[quad1])
        plt.ylabel(labels[quad2])
        plt.title('Wigner Function')
        plt.colorbar(label='Wigner Value')
        plt.tight_layout()
        plt.show()

