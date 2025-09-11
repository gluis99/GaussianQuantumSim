from Gaussian_States import GaussianState
from Gaussian_Gates_Symplectic import *

vacuum_3= GaussianState.vacuum(3)
#initial_state_3.plot_wigner_2d()

EPR_state = GaussianState.vacuum(3).apply_gaussian_gate(One_Mode_Squeeze_N_mode(1, 0,0,3)).apply_gaussian_gate(One_Mode_Squeeze_N_mode(1, 0,1,3)).apply_gaussian_gate(Phase_rotation_N_mode(np.pi/2,1,3)).apply_gaussian_gate(Beam_splitter_N_mode(np.pi/4,0,1,3))
EPR_state_alt = GaussianState.vacuum(3).apply_gaussian_gate(Two_Mode_Squeeze_N_mode(1,0,0,1,3))
EPR_state_alt.plot_wigner_2d(0,2)
EPR_state.plot_wigner_2d(0,2)



