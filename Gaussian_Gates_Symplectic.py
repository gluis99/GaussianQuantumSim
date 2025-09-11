import numpy as np

# Gaussian Transformations as symplectic matrices

#One mode transformations
def One_Mode_Squeeze(r,theta):
    """Single mode squeezing symplectic transformation"""
    S = np.array([[np.cosh(r)-np.sinh(r)*np.cos(theta), -np.sinh(r)*np.sin(theta)],
                  [-np.sinh(r)*np.sin(theta), np.cosh(r)+np.sinh(r)*np.cos(theta)]])
    return S

#Two mode transformations
def Phase_rotation(theta):
    """Single mode phase rotation symplectic transformation"""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R

def Beam_splitter(theta):
    """Two mode beam splitter symplectic transformation"""
    tau = np.cos(theta)**2
    BS = np.array([[np.sqrt(tau), 0, np.sqrt(1-tau), 0],
                   [0, np.sqrt(tau), 0, np.sqrt(1-tau)],
                   [-np.sqrt(1-tau), 0, np.sqrt(tau), 0],
                   [0, -np.sqrt(1-tau), 0, np.sqrt(tau)]])
    return BS

def Two_Mode_Squeeze(r, theta):
    """Two mode squeezing symplectic transformation"""
    S = np.array([[np.cosh(r), 0, np.sinh(r)*np.cos(theta), np.sinh(r)*np.sin(theta)],
                  [0, np.cosh(r), np.sinh(r)*np.sin(theta), -np.sinh(r)*np.cos(theta)],
                  [np.sinh(r)*np.cos(theta), np.sinh(r)*np.sin(theta), np.cosh(r), 0],
                  [np.sinh(r)*np.sin(theta), -np.sinh(r)*np.cos(theta), 0, np.cosh(r)]])
    return S

def Controlled_Z(phi):
    """Two mode controlled-Z gate symplectic transformation"""
    CZ = np.array([[1, 0, 0, 0],
                   [0, 1, phi, 0],
                   [0, 0, 1, 0],
                   [phi, 0, 0, 1]])
    return CZ

#################################################################################
#N mode Gaussian transformations
def One_Mode_Squeeze_N_mode(r, theta, mode, N):
    """N mode squeezing symplectic transformation on specified mode (0 to N-1)"""
    S_single = One_Mode_Squeeze(r, theta)
    S = np.identity(2*N)
    S[2*mode:2*mode+2, 2*mode:2*mode+2] = S_single
    return S

def Phase_rotation_N_mode(theta, mode, N):
    """N mode phase rotation symplectic transformation on specified mode (0 to N-1)"""
    R_single = Phase_rotation(theta)
    R = np.identity(2*N)
    R[2*mode:2*mode+2, 2*mode:2*mode+2] = R_single
    return R

def Beam_splitter_N_mode(theta, mode1, mode2, N):
    """N mode beam splitter symplectic transformation on specified modes (0 to N-1)"""
    BS_single = Beam_splitter(theta)
    BS = np.identity(2*N)
    BS[2*mode1:2*mode1+2, 2*mode1:2*mode1+2] = BS_single[0:2, 0:2]
    BS[2*mode1:2*mode1+2, 2*mode2:2*mode2+2] = BS_single[0:2, 2:4]
    BS[2*mode2:2*mode2+2, 2*mode1:2*mode1+2] = BS_single[2:4, 0:2]
    BS[2*mode2:2*mode2+2, 2*mode2:2*mode2+2] = BS_single[2:4, 2:4]
    return BS

def Controlled_Z_N_mode(phi, mode1, mode2, N):
    """N mode controlled-Z gate symplectic transformation on specified modes (0 to N-1)"""
    CZ_single = Controlled_Z(phi)
    CZ = np.identity(2*N)
    CZ[2*mode1:2*mode1+2, 2*mode1:2*mode1+2] = CZ_single[0:2, 0:2]
    CZ[2*mode1:2*mode1+2, 2*mode2:2*mode2+2] = CZ_single[0:2, 2:4]
    CZ[2*mode2:2*mode2+2, 2*mode1:2*mode1+2] = CZ_single[2:4, 0:2]
    CZ[2*mode2:2*mode2+2, 2*mode2:2*mode2+2] = CZ_single[2:4, 2:4]
    return CZ

def Two_Mode_Squeeze_N_mode(r, theta, mode1, mode2, N):
    """N mode two-mode squeezing symplectic transformation on specified modes (0 to N-1)"""
    S_single = Two_Mode_Squeeze(r, theta)
    S = np.identity(2*N)
    S[2*mode1:2*mode1+2, 2*mode1:2*mode1+2] = S_single[0:2, 0:2]
    S[2*mode1:2*mode1+2, 2*mode2:2*mode2+2] = S_single[0:2, 2:4]
    S[2*mode2:2*mode2+2, 2*mode1:2*mode1+2] = S_single[2:4, 0:2]
    S[2*mode2:2*mode2+2, 2*mode2:2*mode2+2] = S_single[2:4, 2:4]
    return S
