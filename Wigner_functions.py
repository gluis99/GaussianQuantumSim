import numpy as np
import matplotlib.pyplot as plt

s = np.linspace(-5,5,200)
X, P = np.meshgrid(s, s)
xi_1mode = np.column_stack([X.ravel(), P.ravel()])

def xi_N(N, num_points=200, range=5):
    """Generate phase space points for N modes."""
    s = np.linspace(-range, range, num_points)
    grids = np.meshgrid(*([s, s] * N))
    xi = np.column_stack([XP.ravel() for XP in grids])
    return xi


# Covariance matrizes
V_vac_N = lambda N: np.identity(2*N)
r_0_N = lambda N: np.zeros(2*N)
V_sq = lambda r: np.array([[np.exp(-2*r),0],[0,np.exp(2*r)]])
V_sq_gen = lambda r,theta: np.array([[np.cosh(2*r)+np.sinh(2*r)*np.cos(theta), np.sinh(2*r)*np.sin(theta)],
                                    [np.sinh(2*r)*np.sin(theta), np.cosh(2*r)-np.sinh(2*r)*np.cos(theta)]]) 
V_EPR_nu = lambda nu: np.array([[nu,0,np.sqrt(nu**2-1),0],[0,nu,0,-np.sqrt(nu**2-1)],
                       [np.sqrt(nu**2-1),0,nu,0],[0,-np.sqrt(nu**2-1),0,nu]])
V_EPR_r = lambda r: V_EPR_nu(np.cosh(2*r))

def wigner_transform(xi,V,r, N):
    """Wigner transform of a N mode Gaussian state with covariance matrix V and displacement r, at the interval x"""
    xi_disp = xi - r
    exponent = np.einsum('ij,jk,ik->i', xi_disp, np.linalg.inv(V), xi_disp)
    Wigner = 1/((2*np.pi)**N *np.sqrt(np.linalg.det(V)))*np.exp(-exponent/2)
    return Wigner

def plot_wigner(V,r,N, num_points=200, range = 5, title="Wigner function", cmap='RdBu'):
    xi = xi_N(N, num_points, range)
    W = wigner_transform(xi,V,r,N).reshape((num_points,)*2*N)
    
    if N == 1:
        plt.figure(figsize=(6,5))
        plt.contourf(W, levels=100, cmap=cmap, extent=(-range,range,-range,range))
        plt.colorbar(label='Wigner function W(x,p)')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.title(title)
        plt.axis('equal')
        plt.show()
    else:
        print("Plotting is only implemented for N=1 mode")

def plot_3D_wigner_1mode(xi, W, num_points=200, range=5, title="3D Wigner function"):
    """
    Plots the Wigner function as a 3D surface using xi and W values.
    xi: 2D array of shape (num_points**2, 2) for 1 mode
    W: 1D array of Wigner function values, same length as xi
    """
    x = xi[:, 0].reshape((num_points, num_points))
    p = xi[:, 1].reshape((num_points, num_points))
    W = W.reshape((num_points, num_points))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, p, W, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_zlabel('Wigner function')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_wigner_2mode_slice(V, r, vary_indices=(0,2), fixed_values={1:0, 3:0}, num_points=200, 
                            range=5, title="Wigner function slice", cmap='RdBu'):
    """
    Plots a 2D slice of the Wigner function for a 2-mode state by varying two indices and fixing the others.
    vary_indices: tuple of two indices to vary (e.g., (0,2) for x1 and x2)
    fixed_values: dictionary with indices as keys and fixed values as values (e.g., {1:0, 3:0} for p1=0, p2=0)
    """
    quad_names = ['x1', 'p1', 'x2', 'p2']
    label_x = quad_names[vary_indices[0]]
    label_y = quad_names[vary_indices[1]]

    s = np.linspace(-range, range, num_points)
    X, Y = np.meshgrid(s, s)
    xi_slice = np.zeros((num_points**2, 4))
    
    # Set the varying indices
    xi_slice[:, vary_indices[0]] = X.ravel()
    xi_slice[:, vary_indices[1]] = Y.ravel()
    
    # Set the fixed indices
    for idx, val in fixed_values.items():
        xi_slice[:, idx] = val
    
    W = wigner_transform(xi_slice, V, r, N=2).reshape((num_points, num_points))
    
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, W, levels=100, cmap=cmap)
    plt.colorbar(label='Wigner function W')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_wigner_3D_2mode_slice(V, r, vary_indices=(0,2), fixed_values={1:0, 3:0}, num_points=100, 
                            range=5, title="3D Wigner function slice"):
    """
    Plots a 3D surface of a slice of the Wigner function for a 2-mode state by varying two indices and fixing the others.
    vary_indices: tuple of two indices to vary (e.g., (0,2) for x1 and x2)
    fixed_values: dictionary with indices as keys and fixed values as values (e.g., {1:0, 3:0} for p1=0, p2=0)
    """
    quad_names = ['x1', 'p1', 'x2', 'p2']
    label_x = quad_names[vary_indices[0]]
    label_y = quad_names[vary_indices[1]]

    s = np.linspace(-range, range, num_points)
    X, Y = np.meshgrid(s, s)
    xi_slice = np.zeros((num_points**2, 4))
    
    # Set the varying indices
    xi_slice[:, vary_indices[0]] = X.ravel()
    xi_slice[:, vary_indices[1]] = Y.ravel()
    
    # Set the fixed indices
    for idx, val in fixed_values.items():
        xi_slice[:, idx] = val
    
    W = wigner_transform(xi_slice, V, r, N=2)

    # Reshape for plotting
    X = X.reshape((num_points, num_points))
    Y = Y.reshape((num_points, num_points))
    W = W.reshape((num_points, num_points))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, W, cmap='viridis', edgecolor='none')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel('Wigner function')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

V = V_EPR_r(0.5)
r = r_0_N(2)

plot_wigner_3D_2mode_slice(V, r, vary_indices=(1,3), fixed_values={0:0, 2:0},
                          num_points=100, range=10)
