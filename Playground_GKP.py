from Gaussian_States import GaussianState
from States import (
	GaussianState,
	CatState,
	GKPState,
	GKPFiguresOfMerit,
	characteristic_function_from_wigner_1mode,
)
from Gaussian_Gates_Symplectic import *
import numpy as np
import matplotlib.pyplot as plt

r = lambda r_dB: 10**(r_dB/20)

# Build GKP grid from symplectic transformation S, d
def plot_gkp_grid(S, d, range = (-10, 10), N_points = 1001):
	# Create a grid of points in phase space
	x = np.linspace(range[0], range[1], N_points)
	p = np.linspace(range[0], range[1], N_points)
	X, P = np.meshgrid(x, p)
	
	# Get GKP wigner function from symplectic transformation
	gkp_state = GKPState.from_stabilizers(stabilizers=S, logical_displacement=d, approximation='ideal', approximation_params={})
	W = gkp_state.plot_wigner_2d(rng=range, num_points=N_points, show=False)
	# Plot the GKP grid
	vmax = 1/np.pi
	plt.imshow(W, extent=(range[0], range[1], range[0], range[1]), cmap='RdBu_r', vmin=-vmax, vmax=vmax)

	plt.colorbar(label='Wigner Function Value')
	plt.title('GKP Grid in Phase Space')
	plt.xlabel('x')
	plt.ylabel('p')
	plt.grid()
	plt.show()

S_1 = lambda theta_1, r, theta_2: Phase_rotation(theta_1) @ One_Mode_Squeeze(r,0) @ Phase_rotation(theta_2)
d_1 = np.array([0, 0])

for theta1, theta2 in zip([0, np.pi/4, np.pi/2], [0, np.pi/4, np.pi/2]):
	plot_gkp_grid(S=S_1(theta1,0.25,theta2), d=d_1, range=(-10, 10), N_points=1001)



