import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from stl import mesh


def createNeighbours(X):
    """
    This function creates matrices containing the positions of the neighbouring points for every point in the net, as required by the mass-spring model
    :param X: a position matrix of the net
    :return: the north, south, east, west, diagonal and bending moment neighbours
    """

    XN = np.roll(X, -1, 0)
    XS = np.roll(X, 1, 0)
    XE = np.roll(X, 1, 1)
    XW = np.roll(X, -1, 1)
    XNN = np.roll(X, -2, 0)
    XSS = np.roll(X, 2, 0)
    XEE = np.roll(X, 2, 1)
    XWW = np.roll(X, -2, 1)
    XNE = np.roll(XN, 1, 1)
    XNW = np.roll(XN, -1, 1)
    XSE = np.roll(XS, 1, 1)
    XSW = np.roll(XS, -1, 1)
    return XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW


def Fe(X, const):
    Force = np.zeros(X.shape)
    sgn = np.sign(X)
    for i in range(len(const)):
        Force += const[i] * np.abs(X) ** (i + 1)
    Force *= sgn
    return Force


class Canopy_Material:

    def __init__(self, E_fibre, surface_density, elasticity_ax, elasticity_diag, fibre_diameter, porosity):
        self.E_fibre = E_fibre
        self.rho = surface_density
        self.ElAx = elasticity_ax
        self.ElDiag = elasticity_diag
        self.D_fibre = fibre_diameter
        self.porosity = porosity


class Suspension_Material:

    def __init__(self, E_fibre, line_density, elasticity_ax, line_diameter):
        self.E_fibre = E_fibre
        self.rho = line_density
        self.ElAx = elasticity_ax
        self.D_line = line_diameter


class Canopy:
    """
    This class creates the mass-spring system used for discretizing the fishing net at an arbitrary resolution. It also simulates its deformation
    using the Euler time integration method
    """

    def __init__(self, X, Y, Z, canopy_material, reinforcement_material, suspension_material, reinforcement_width, num_sus, num_gores, dp):

        """
        This constructor initializes the spring constants of the mass-spring system based on the E-modulus and geometry of the fishing net.
        It also initializes several other parameters of the water flow.
        :param X: the matrix containing the X-positions of the virtual nodes
        :param Y: the matrix containing the Y-positions of the virtual nodes
        :param Z: the matrix containing the Z-positions of the virtual nodes
        :param E_fibre: the E-modulus of the fibre used in twines (usually Nylon)
        :param twine_length: the length of one twine (distance between two knots)
        :param twine_density: the density of the material used in the twine
        :param knot_mass: the mass of the knots
        :param d_twine: the twine diameter
        :param Sn: the solidity ratio of the fishing net
        :param V_inf: the free-stream velocity of the fishing net
        :param rho: the density of the fluid in which the fishing net is deployed
        :param scale: the scaling factor of the fishing net geometry (used to speed up the simulation)
        """

        self.E_fabric = canopy_material.E_fibre * np.pi / 4
        E_fabric = self.E_fabric
        self.DFiber = canopy_material.D_fibre
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Vx = np.zeros(X.shape)
        self.Vy = np.zeros(X.shape)
        self.Vz = np.zeros(X.shape)
        self.porosity = canopy_material.porosity
        self.NumSus = num_sus
        self.NumGores = num_gores
        self.Color = np.zeros(self.X.shape)
        self.Nx, self.Ny = X.shape
        self.RWidth = reinforcement_width
        self.ECoeffAx = canopy_material.ElAx
        self.ECoeffDiag = canopy_material.ElDiag
        self.ECoeffReinf = reinforcement_material.ElAx
        self.ECoeffReinfDiag = reinforcement_material.ElDiag
        self.ECoeffSus = suspension_material.ElAx
        self.sus_rho = suspension_material.rho
        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(self.X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(self.Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(self.Z)

        dXN, dXS, dXE, dXW = XN - X, XS - X, XE - X, XW - X
        dYN, dYS, dYE, dYW = YN - Y, YS - Y, YE - Y, YW - Y
        dZN, dZS, dZE, dZW = ZN - Z, ZS - Z, ZE - Z, ZW - Z
        dXNE, dXNW, dXSE, dXSW = XNE - X, XNW - X, XSE - X, XSW - X
        dYNE, dYNW, dYSE, dYSW = YNE - Y, YNW - Y, YSE - Y, YSW - Y
        dZNE, dZNW, dZSE, dZSW = ZNE - Z, ZNW - Z, ZSE - Z, ZSW - Z
        dXNN, dXSS, dXEE, dXWW = XNN - X, XSS - X, XEE - X, XWW - X
        dYNN, dYSS, dYEE, dYWW = YNN - Y, YSS - Y, YEE - Y, YWW - Y
        dZNN, dZSS, dZEE, dZWW = ZNN - Z, ZSS - Z, ZEE - Z, ZWW - Z

        self.LN0 = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
        self.LS0 = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
        self.LE0 = (dXE ** 2 + dYE ** 2 + dZE ** 2) ** 0.5
        self.LW0 = (dXW ** 2 + dYW ** 2 + dZW ** 2) ** 0.5
        self.LNE0 = (dXNE ** 2 + dYNE ** 2 + dZNE ** 2) ** 0.5
        self.LNW0 = (dXNW ** 2 + dYNW ** 2 + dZNW ** 2) ** 0.5
        self.LSE0 = (dXSE ** 2 + dYSE ** 2 + dZSE ** 2) ** 0.5
        self.LSW0 = (dXSW ** 2 + dYSW ** 2 + dZSW ** 2) ** 0.5
        self.LNN0 = (dXNN ** 2 + dYNN ** 2 + dZNN ** 2) ** 0.5
        self.LSS0 = (dXSS ** 2 + dYSS ** 2 + dZSS ** 2) ** 0.5
        self.LEE0 = (dXEE ** 2 + dYEE ** 2 + dZEE ** 2) ** 0.5
        self.LWW0 = (dXWW ** 2 + dYWW ** 2 + dZWW ** 2) ** 0.5

        self.LNE0[-1, :] = self.LNE0[-2, :]
        self.LNW0[-1, :] = self.LNW0[-2, :]
        self.LSE0[0, :] = self.LSE0[1, :]
        self.LSW0[0, :] = self.LSW0[1, :]
        self.LN0[-1, :] = self.LN0[-2, :]
        self.LS0[0, :] = self.LS0[1, :]
        self.LNN0[-2, :] = self.LNN0[-3, :]
        self.LNN0[-1, :] = self.LNN0[-2, :]
        self.LSS0[1, :] = self.LSS0[2, :]
        self.LSS0[0, :] = self.LSS0[1, :]

        self.kNW = np.zeros(self.X.shape)
        self.kNE = np.zeros(self.X.shape)
        self.kSW = np.zeros(self.X.shape)
        self.kSE = np.zeros(self.X.shape)
        self.kNN = np.ones(self.X.shape) * 0
        self.kSS = np.ones(self.X.shape) * 0
        self.kEE = np.ones(self.X.shape) * 0
        self.kWW = np.ones(self.X.shape) * 0
        self.kNW[-1, :] = self.kNE[-1, :] = 0
        self.kSW[0, :] = self.kSE[0, :] = 0
        self.kNN[-2:-1, :] = self.kSS[0:1, :] = 0
        self.rho = canopy_material.rho
        self.reinf_rho = reinforcement_material.rho
        self.dP = np.ones(X.shape) * dp
        RNumWidthNS = 1
        if np.average((self.LE0 + self.LW0) * 0.5) != 0:
            RNumWidthNS = max(int(self.RWidth / (np.average((self.LE0 + self.LW0) * 0.5))), 1)

        self.M = 0.5 * (self.LN0 + self.LS0) * 0.5 * (self.LE0 + self.LW0) * canopy_material.rho * np.pi / 4 * self.porosity
        self.M[:RNumWidthNS, :] += (0.5 * (self.LN0 + self.LS0) * 0.5 * (self.LE0 + self.LW0) * self.reinf_rho * np.pi / 4)[:RNumWidthNS, :]
        self.M[-RNumWidthNS:, :] += (0.5 * (self.LN0 + self.LS0) * 0.5 * (self.LE0 + self.LW0) * self.reinf_rho * np.pi / 4)[-RNumWidthNS:, :]
        self.computeNormals()

    def computeNormals(self):
        X = self.X
        Y = self.Y
        Z = self.Z

        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)

        XN2 = 0.5 * (XN + X)
        YN2 = 0.5 * (YN + Y)
        ZN2 = 0.5 * (ZN + Z)
        XS2 = 0.5 * (XS + X)
        YS2 = 0.5 * (YS + Y)
        ZS2 = 0.5 * (ZS + Z)
        XE2 = 0.5 * (XE + X)
        YE2 = 0.5 * (YE + Y)
        ZE2 = 0.5 * (ZE + Z)
        XW2 = 0.5 * (XW + X)
        YW2 = 0.5 * (YW + Y)
        ZW2 = 0.5 * (ZW + Z)

        dXNS = XN2 - XS2
        dYNS = YN2 - YS2
        dZNS = ZN2 - ZS2
        dXWE = XW2 - XE2
        dYWE = YW2 - YE2
        dZWE = ZW2 - ZE2

        norm = (dXNS ** 2 + dYNS ** 2 + dZNS ** 2) ** 0.5
        norm = np.where(norm == 0, 1, norm)

        dXNS = dXNS / norm
        dYNS = dYNS / norm
        dZNS = dZNS / norm

        norm = (dXWE ** 2 + dYWE ** 2 + dZWE ** 2) ** 0.5
        norm = np.where(norm == 0, 1, norm)

        dXWE = dXWE / norm
        dYWE = dYWE / norm
        dZWE = dZWE / norm

        nX = dYNS * dZWE - dZNS * dYWE
        nY = dZNS * dXWE - dXNS * dZWE
        nZ = dXNS * dYWE - dYNS * dXWE

        nX[-1] = nX[-2]
        nY[-1] = nY[-2]
        nZ[-1] = nZ[-2]
        nX[0] = nX[1]
        nY[0] = nY[1]
        nZ[0] = nZ[1]

        self.nX = nX
        self.nY = nY
        self.nZ = nZ

    def Sp(self):

        """
        This function calculates the hydrodynamic forces on the discretized fishing net based on the pressure difference induced by the water flow
        :return: the normal-acting hydrodynamic forces on each point in the net (in the drag parameter)
        """

        X = self.X
        Y = self.Y
        Z = self.Z

        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)

        XN2 = 0.5 * (XN + X)
        YN2 = 0.5 * (YN + Y)
        ZN2 = 0.5 * (ZN + Z)
        XS2 = 0.5 * (XS + X)
        YS2 = 0.5 * (YS + Y)
        ZS2 = 0.5 * (ZS + Z)
        XE2 = 0.5 * (XE + X)
        YE2 = 0.5 * (YE + Y)
        ZE2 = 0.5 * (ZE + Z)
        XW2 = 0.5 * (XW + X)
        YW2 = 0.5 * (YW + Y)
        ZW2 = 0.5 * (ZW + Z)

        dXNS = XN2 - XS2
        dYNS = YN2 - YS2
        dZNS = ZN2 - ZS2
        dXWE = XW2 - XE2
        dYWE = YW2 - YE2
        dZWE = ZW2 - ZE2

        norm = (dXNS ** 2 + dYNS ** 2 + dZNS ** 2) ** 0.5
        norm = np.where(norm == 0, 1, norm)

        dXNS = dXNS / norm
        dYNS = dYNS / norm
        dZNS = dZNS / norm

        norm = (dXWE ** 2 + dYWE ** 2 + dZWE ** 2) ** 0.5
        norm = np.where(norm == 0, 1, norm)

        dXWE = dXWE / norm
        dYWE = dYWE / norm
        dZWE = dZWE / norm

        nX = dYNS * dZWE - dZNS * dYWE
        nY = dZNS * dXWE - dXNS * dZWE
        nZ = dXNS * dYWE - dYNS * dXWE

        nX[-1] = nX[-2]
        nY[-1] = nY[-2]
        nZ[-1] = nZ[-2]
        nX[0] = nX[1]
        nY[0] = nY[1]
        nZ[0] = nZ[1]

        self.nX = nX
        self.nY = nY
        self.nZ = nZ

        dXN, dXS, dXE, dXW = XN - X, XS - X, XE - X, XW - X
        dYN, dYS, dYE, dYW = YN - Y, YS - Y, YE - Y, YW - Y
        dZN, dZS, dZE, dZW = ZN - Z, ZS - Z, ZE - Z, ZW - Z

        LN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
        LS = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
        LE = (dXE ** 2 + dYE ** 2 + dZE ** 2) ** 0.5
        LW = (dXW ** 2 + dYW ** 2 + dZW ** 2) ** 0.5

        S = 0.5 * (LN + LS) * 0.5 * (LE + LW)
        S[-1, :] = S[-2, :]
        S[0, :] = S[1, :]

        rho = self.rho

        Fx = S * self.dP * nX
        Fy = S * self.dP * nY
        Fz = S * self.dP * nZ

        return Fx, Fy, Fz, 0, 0, 0

    def getAcc(self, Fixed, dt):

        """
        This function computes the acceleration and performs the forward Euler time integration of every point in the discretized net
        :param X: the X-positions of the points in the net
        :param Y: the Y-positions of the points in the net
        :param Z: the Z-positions of the points in the net
        :param Vx: the X-velocities of the points in the net
        :param Vy: the Y-velocities of the points in the net
        :param Vz: the Z-velocities of the points in the net
        :param Fixed: the matrix of net points that should be fixed throughout the simulation
        :param dt: the time step
        :return: the new position and velocity arrays after one time step
        """
        X = self.X
        Y = self.Y
        Z = self.Z
        Vx = self.Vx
        Vy = self.Vy
        Vz = self.Vz
        g0 = 0
        gz = np.ones(self.M.shape) * g0
        C = 1 / X.shape[0] / X.shape[1]

        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)

        dXN, dXS, dXE, dXW = XN - X, XS - X, XE - X, XW - X
        dYN, dYS, dYE, dYW = YN - Y, YS - Y, YE - Y, YW - Y
        dZN, dZS, dZE, dZW = ZN - Z, ZS - Z, ZE - Z, ZW - Z
        dXNE, dXNW, dXSE, dXSW = XNE - X, XNW - X, XSE - X, XSW - X
        dYNE, dYNW, dYSE, dYSW = YNE - Y, YNW - Y, YSE - Y, YSW - Y
        dZNE, dZNW, dZSE, dZSW = ZNE - Z, ZNW - Z, ZSE - Z, ZSW - Z
        dXNN, dXSS, dXEE, dXWW = XNN - X, XSS - X, XEE - X, XWW - X
        dYNN, dYSS, dYEE, dYWW = YNN - Y, YSS - Y, YEE - Y, YWW - Y
        dZNN, dZSS, dZEE, dZWW = ZNN - Z, ZSS - Z, ZEE - Z, ZWW - Z

        LN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
        LS = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
        LE = (dXE ** 2 + dYE ** 2 + dZE ** 2) ** 0.5
        LW = (dXW ** 2 + dYW ** 2 + dZW ** 2) ** 0.5
        LNE = (dXNE ** 2 + dYNE ** 2 + dZNE ** 2) ** 0.5
        LNW = (dXNW ** 2 + dYNW ** 2 + dZNW ** 2) ** 0.5
        LSE = (dXSE ** 2 + dYSE ** 2 + dZSE ** 2) ** 0.5
        LSW = (dXSW ** 2 + dYSW ** 2 + dZSW ** 2) ** 0.5
        LNN = (dXNN ** 2 + dYNN ** 2 + dZNN ** 2) ** 0.5
        LSS = (dXSS ** 2 + dYSS ** 2 + dZSS ** 2) ** 0.5
        LEE = (dXEE ** 2 + dYEE ** 2 + dZEE ** 2) ** 0.5
        LWW = (dXWW ** 2 + dYWW ** 2 + dZWW ** 2) ** 0.5

        LN = np.where(LN == 0, 1, LN)
        LS = np.where(LS == 0, 1, LS)
        LE = np.where(LE == 0, 1, LE)
        LW = np.where(LW == 0, 1, LW)
        LNE = np.where(LNE == 0, 1, LNE)
        LNW = np.where(LNW == 0, 1, LNW)
        LSE = np.where(LSE == 0, 1, LSE)
        LSW = np.where(LSW == 0, 1, LSW)
        LNN = np.where(LNN == 0, 1, LNN)
        LSS = np.where(LSS == 0, 1, LSS)
        LEE = np.where(LEE == 0, 1, LEE)
        LWW = np.where(LWW == 0, 1, LWW)

        FN = Fe(LN - self.LN0, self.ECoeffAx) * 0.5 * (self.LE0 + self.LW0) / self.LN0
        FS = Fe(LS - self.LS0, self.ECoeffAx) * 0.5 * (self.LE0 + self.LW0) / self.LS0
        FE = Fe(LE - self.LE0, self.ECoeffAx) * 0.5 * (self.LN0 + self.LS0) / self.LE0
        FW = Fe(LW - self.LW0, self.ECoeffAx) * 0.5 * (self.LN0 + self.LS0) / self.LW0

        FNE = Fe(LNE - self.LNE0, self.ECoeffDiag) * 0.5 * (self.LNW0 + self.LSE0) / self.LNE0
        FNW = Fe(LNW - self.LNW0, self.ECoeffDiag) * 0.5 * (self.LNE0 + self.LSW0) / self.LNW0
        FSE = Fe(LSE - self.LSE0, self.ECoeffDiag) * 0.5 * (self.LNE0 + self.LSW0) / self.LSE0
        FSW = Fe(LSW - self.LSW0, self.ECoeffDiag) * 0.5 * (self.LNW0 + self.LSE0) / self.LSW0
        FNN = Fe((LNN - self.LNN0), self.ECoeffReinf) * ((self.LE0 + self.LW0) / self.LNN0) / 100
        FSS = Fe((LSS - self.LSS0), self.ECoeffReinf) * ((self.LE0 + self.LW0) / self.LSS0) / 100
        FEE = Fe((LEE - self.LEE0), self.ECoeffReinf) * ((self.LN0 + self.LS0) / self.LEE0) / 100
        FWW = Fe((LWW - self.LWW0), self.ECoeffReinf) * ((self.LN0 + self.LS0) / self.LWW0) / 100

        RNumWidthNS = max(int(self.RWidth / (np.average((self.LE0 + self.LW0) * 0.5))), 1)
        RNumWidthEW = max(int(self.RWidth / (np.average((self.LN0 + self.LS0) * 0.5))), 1)

        FN[-RNumWidthNS:, :] += Fe((LN - self.LN0)[-RNumWidthNS:, :], self.ECoeffReinf) * (0.5 * (self.LE0 + self.LW0) / self.LN0)[-RNumWidthNS:, :]
        FN[:RNumWidthNS, :] += Fe((LN - self.LN0)[:RNumWidthNS, :], self.ECoeffReinf) * (0.5 * (self.LE0 + self.LW0) / self.LN0)[:RNumWidthNS, :]
        FS[-RNumWidthNS:, :] += Fe((LS - self.LS0)[-RNumWidthNS:, :], self.ECoeffReinf) * (0.5 * (self.LE0 + self.LW0) / self.LS0)[-RNumWidthNS:, :]
        FS[:RNumWidthNS, :] += Fe((LS - self.LS0)[:RNumWidthNS, :], self.ECoeffReinf) * (0.5 * (self.LE0 + self.LW0) / self.LS0)[:RNumWidthNS, :]
        FE[:RNumWidthNS, :] += Fe((LE - self.LE0)[:RNumWidthNS, :], self.ECoeffReinf) * (0.5 * (self.LN0 + self.LS0) / self.LE0)[:RNumWidthNS, :]
        FE[-RNumWidthNS:, :] += Fe((LE - self.LE0)[-RNumWidthNS:, :], self.ECoeffReinf) * (0.5 * (self.LN0 + self.LS0) / self.LE0)[-RNumWidthNS:, :]
        FW[:RNumWidthNS, :] += Fe((LW - self.LW0)[:RNumWidthNS, :], self.ECoeffReinf) * (0.5 * (self.LN0 + self.LS0) / self.LW0)[:RNumWidthNS, :]
        FW[-RNumWidthNS:, :] += Fe((LW - self.LW0)[-RNumWidthNS:, :], self.ECoeffReinf) * (0.5 * (self.LN0 + self.LS0) / self.LW0)[-RNumWidthNS:, :]
        FNE[-RNumWidthNS:, :] += Fe((LNE - self.LNE0)[-RNumWidthNS:, :], self.ECoeffReinfDiag) * (0.5 * (self.LNW0 + self.LSE0) / self.LNE0)[-RNumWidthNS:, :]
        FNE[:RNumWidthNS, :] += Fe((LNE - self.LNE0)[:RNumWidthNS, :], self.ECoeffReinfDiag) * (0.5 * (self.LNW0 + self.LSE0) / self.LNE0)[:RNumWidthNS, :]
        FNW[-RNumWidthNS:, :] += Fe((LNW - self.LNW0)[-RNumWidthNS:, :], self.ECoeffReinfDiag) * (0.5 * (self.LNE0 + self.LSW0) / self.LNW0)[-RNumWidthNS:, :]
        FNW[:RNumWidthNS, :] += Fe((LNW - self.LNW0)[:RNumWidthNS, :], self.ECoeffReinfDiag) * (0.5 * (self.LNE0 + self.LSW0) / self.LNW0)[:RNumWidthNS, :]
        FSE[-RNumWidthNS:, :] += Fe((LSE - self.LSE0)[-RNumWidthNS:, :], self.ECoeffReinfDiag) * (0.5 * (self.LSW0 + self.LNE0) / self.LSE0)[-RNumWidthNS:, :]
        FSE[:RNumWidthNS, :] += Fe((LSE - self.LSE0)[:RNumWidthNS, :], self.ECoeffReinfDiag) * (0.5 * (self.LSW0 + self.LNE0) / self.LSE0)[:RNumWidthNS, :]
        FSW[-RNumWidthNS:, :] += Fe((LSW - self.LSW0)[-RNumWidthNS:, :], self.ECoeffReinfDiag) * (0.5 * (self.LNW0 + self.LSE0) / self.LSW0)[-RNumWidthNS:, :]
        FSW[:RNumWidthNS, :] += Fe((LSW - self.LSW0)[:RNumWidthNS, :], self.ECoeffReinfDiag) * (0.5 * (self.LNW0 + self.LSE0) / self.LSW0)[:RNumWidthNS, :]
        FEE[:RNumWidthNS, :] += Fe((LEE - self.LEE0)[:RNumWidthNS, :], self.ECoeffReinf) * ((self.LN0 + self.LS0) / self.LEE0)[:RNumWidthNS, :] / 50
        FEE[-RNumWidthNS:, :] += Fe((LEE - self.LEE0)[-RNumWidthNS:, :], self.ECoeffReinf) * ((self.LN0 + self.LS0) / self.LEE0)[-RNumWidthNS:, :] / 50
        FWW[:RNumWidthNS, :] += Fe((LWW - self.LWW0)[:RNumWidthNS, :], self.ECoeffReinf) * ((self.LN0 + self.LS0) / self.LWW0)[:RNumWidthNS, :] / 50
        FWW[-RNumWidthNS:, :] += Fe((LWW - self.LWW0)[-RNumWidthNS:, :], self.ECoeffReinf) * ((self.LN0 + self.LS0) / self.LWW0)[-RNumWidthNS:, :] / 50
        FNN[:RNumWidthNS, :] += Fe((LNN - self.LNN0)[:RNumWidthNS, :], self.ECoeffReinf) * ((self.LE0 + self.LW0) / self.LNN0)[:RNumWidthNS, :] / 50
        FNN[-RNumWidthNS:, :] += Fe((LNN - self.LNN0)[-RNumWidthNS:, :], self.ECoeffReinf) * ((self.LE0 + self.LW0) / self.LSS0)[-RNumWidthNS:, :] / 50
        FSS[:RNumWidthNS, :] += Fe((LSS - self.LSS0)[:RNumWidthNS, :], self.ECoeffReinf) * ((self.LE0 + self.LW0) / self.LWW0)[:RNumWidthNS, :] / 50
        FSS[-RNumWidthNS:, :] += Fe((LSS - self.LSS0)[-RNumWidthNS:, :], self.ECoeffReinf) * ((self.LE0 + self.LW0) / self.LWW0)[-RNumWidthNS:, :] / 50

        self.Color[-RNumWidthNS:, :] = 1
        self.Color[:RNumWidthNS, :] = 1

        Sus_Index = np.linspace(0, len(X[0]), self.NumSus + 1)
        Sus_Index = np.array(Sus_Index[:-1], int)
        offset = int(self.NumSus / self.NumGores)
        Gore_Index = Sus_Index[::offset]
        # print(Gore_Index)
        for i in Gore_Index:
            im = int(i - RNumWidthEW / 2) % len(FE[0])
            ip = int(i + RNumWidthEW / 2) % len(FE[0])
            if im > ip:
                im -= len(FE[0])
            FE[:, im:ip] += (Fe(LE - self.LE0, self.ECoeffReinf) * 0.5 * (self.LN0 + self.LS0) / self.LE0)[:, im:ip]
            FW[:, im:ip] += (Fe(LW - self.LW0, self.ECoeffReinf) * 0.5 * (self.LN0 + self.LS0) / self.LW0)[:, im:ip]
            FN[:, im:ip] += (Fe(LN - self.LN0, self.ECoeffReinf) * 0.5 * (self.LE0 + self.LW0) / self.LN0)[:, im:ip]
            FS[:, im:ip] += (Fe(LS - self.LS0, self.ECoeffReinf) * 0.5 * (self.LE0 + self.LW0) / self.LS0)[:, im:ip]
            FNE[:, im:ip] += (Fe(LNE - self.LNE0, self.ECoeffReinfDiag) * 0.5 * (self.LNW0 + self.LSE0) / self.LNE0)[:, im:ip]
            FNW[:, im:ip] += (Fe(LNW - self.LNW0, self.ECoeffReinfDiag) * 0.5 * (self.LNE0 + self.LSW0) / self.LNW0)[:, im:ip]
            FSE[:, im:ip] += (Fe(LSE - self.LSE0, self.ECoeffReinfDiag) * 0.5 * (self.LNE0 + self.LSW0) / self.LSE0)[:, im:ip]
            FSW[:, im:ip] += (Fe(LSW - self.LSW0, self.ECoeffReinfDiag) * 0.5 * (self.LSE0 + self.LNW0) / self.LSW0)[:, im:ip]
            FNN[:, im:ip] += (Fe(LNN - self.LNN0, self.ECoeffReinf) * (self.LEE0 + self.LWW0) / self.LNN0)[:, im:ip] / 200
            FSS[:, im:ip] += (Fe(LSS - self.LSS0, self.ECoeffReinf) * (self.LEE0 + self.LWW0) / self.LSS0)[:, im:ip] / 200
            FEE[:, im:ip] += (Fe(LEE - self.LEE0, self.ECoeffReinf) * (self.LNN0 + self.LSS0) / self.LEE0)[:, im:ip] / 200
            FWW[:, im:ip] += (Fe(LWW - self.LWW0, self.ECoeffReinf) * (self.LNN0 + self.LSS0) / self.LWW0)[:, im:ip] / 200
            FN[:, i] += (Fe(LN - self.LN0, self.ECoeffSus) * 0.5 * (self.LE0 + self.LW0) / self.LN0)[:, i]
            FS[:, i] += (Fe(LS - self.LS0, self.ECoeffSus) * 0.5 * (self.LE0 + self.LW0) / self.LS0)[:, i]
            self.Color[:, im:ip] = 1

        FN[-1, :] = 0
        FNE[-1, :] = 0
        FNW[-1, :] = 0
        FS[0, :] = 0
        FSE[0, :] = 0
        FSW[0, :] = 0
        FNN[-1, :] = 0
        FNN[-2, :] = 0
        FSS[0, :] = 0
        FSS[1, :] = 0

        Dx, Dy, Dz, Lx, Ly, Lz = self.Sp()
        Ax = 1 / self.M * (FN * dXN / LN + FS * dXS / LS + FE * dXE / LE + FW * dXW / LW +
                           FNE * dXNE / LNE + FNW * dXNW / LNW +
                           FSE * dXSE / LSE + FSW * dXSW / LSW +
                           FNN * dXNN / LNN + FSS * dXSS / LSS +
                           FEE * dXEE / LEE + FWW * dXWW / LWW - C * Vx * np.abs(Vx) * Fixed + Dx + Lx)
        Ay = 1 / self.M * (FN * dYN / LN + FS * dYS / LS + FE * dYE / LE + FW * dYW / LW +
                           FNE * dYNE / LNE + FNW * dYNW / LNW +
                           FSE * dYSE / LSE + FSW * dYSW / LSW +
                           FNN * dYNN / LNN + FSS * dYSS / LSS +
                           FEE * dYEE / LEE + FWW * dYWW / LWW - C * Vy * np.abs(Vy) * Fixed + Dy + Ly)
        Az = 1 / self.M * (FN * dZN / LN + FS * dZS / LS + FE * dZE / LE + FW * dZW / LW +
                           FNE * dZNE / LNE + FNW * dZNW / LNW +
                           FSE * dZSE / LSE + FSW * dZSW / LSW +
                           FNN * dZNN / LNN + FSS * dZSS / LSS +
                           FEE * dZEE / LEE + FWW * dZWW / LWW - C * Vz * np.abs(Vz) * Fixed + Dz + Lz) + gz

        return Ax, Ay, Az

    def plot(self, ax, color, scamap):

        """
        This function plots the discretized fishing net in 3D for every 1000 time steps
        :param frame: the current time step
        :return: N/A
        """

        X = self.X
        Y = self.Y
        Z = self.Z
        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(self.X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(self.Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(self.Z)

        dXN, dXS, dXE, dXW = XN - X, XS - X, XE - X, XW - X
        dYN, dYS, dYE, dYW = YN - Y, YS - Y, YE - Y, YW - Y
        dZN, dZS, dZE, dZW = ZN - Z, ZS - Z, ZE - Z, ZW - Z
        for i in range(len(X)):
            if self.Color[i][0] == 1:
                ax.plot3D(X[i], Y[i], Z[i], 'red', linewidth=0.25)
            else:
                ax.plot3D(X[i], Y[i], Z[i], color, linewidth=0.25)
        for i in range(len(X.T)):
            ax.plot3D(X.T[i], Y.T[i], Z.T[i], color, linewidth=0.25)

        dN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
        dS = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
        dE = (dXE ** 2 + dYE ** 2 + dZE ** 2) ** 0.5
        dW = (dXW ** 2 + dYW ** 2 + dZW ** 2) ** 0.5
        dNS = 0.5 * (dN + dS)
        dEW = 0.5 * (dE + dW)
        StrainNS = dNS - 0.5 * (self.LN0 + self.LS0)
        StrainEW = dEW - 0.5 * (self.LE0 + self.LW0)
        Strain = (StrainNS ** 2 + StrainEW ** 2) ** 0.5
        FeNS = Fe(StrainNS, self.ECoeffAx) / 0.5 / (self.LN0 + self.LS0)
        FeEW = Fe(StrainEW, self.ECoeffAx) / 0.5 / (self.LE0 + self.LW0)
        FeTot = Fe(Strain, self.ECoeffAx)
        StressNS = FeNS / self.DFiber
        StressNS[0, :] = StressNS[1, :].copy()
        StressNS[-1, :] = StressNS[-2, :].copy()
        StressEW = FeEW / self.DFiber
        StressEW[0, :] = StressEW[1, :].copy()
        StressEW[-1, :] = StressEW[-2, :].copy()

        StressVM = ((StressEW - StressNS)**2 * 0.5)**0.5

        fcolors = scamap.to_rgba(StressVM / 10 ** 6)
        surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, lw=0.6, antialiased=True, vmin=np.min(StressVM), vmax=np.max(StressVM))

        # for i in range(0, len(X), 2):
        #     for j in range(0, len(X[0]), 2):
        #         ax.plot3D([X[i][j], X[i][j] + self.nX[i][j]*0.1], [Y[i][j], Y[i][j] + self.nY[i][j]*0.1], [Z[i][j], Z[i][j] + self.nZ[i][j]*0.1], 'red', linewidth = 0.4)

    def saveNormals(self):

        """
        This function records and saves in a CSV file the normal vectors to the surface of the net for each of its points
        :return: N/A
        """

        X = self.X
        Y = self.Y
        Z = self.Z

        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)

        XN2 = 0.5 * (XN + X)
        YN2 = 0.5 * (YN + Y)
        ZN2 = 0.5 * (ZN + Z)
        XS2 = 0.5 * (XS + X)
        YS2 = 0.5 * (YS + Y)
        ZS2 = 0.5 * (ZS + Z)
        XE2 = 0.5 * (XE + X)
        YE2 = 0.5 * (YE + Y)
        ZE2 = 0.5 * (ZE + Z)
        XW2 = 0.5 * (XW + X)
        YW2 = 0.5 * (YW + Y)
        ZW2 = 0.5 * (ZW + Z)

        dXNS = XN2 - XS2
        dYNS = YN2 - YS2
        dZNS = ZN2 - ZS2
        dXWE = XW2 - XE2
        dYWE = YW2 - YE2
        dZWE = ZW2 - ZE2

        norm = (dXNS ** 2 + dYNS ** 2 + dZNS ** 2) ** 0.5

        dXNS = dXNS / norm
        dYNS = dYNS / norm
        dZNS = dZNS / norm

        norm = (dXWE ** 2 + dYWE ** 2 + dZWE ** 2) ** 0.5

        dXWE = dXWE / norm
        dYWE = dYWE / norm
        dZWE = dZWE / norm

        Matrix = np.array(
            [self.nX[:-1, :-1].flatten(), self.nY[:-1, :-1].flatten(), self.nZ[:-1, :-1].flatten(), dXNS[:-1, :-1].flatten(), dYNS[:-1, :-1].flatten(), dZNS[:-1, :-1].flatten()]).T
        Titles = ["e1x", "e1y", "e1z", "e2x", "e2y", "e2z"]
        self.saveCSV("Normals", Titles, Matrix, True)

    def saveCSV(self, name, Titles, Matrix, withTitles):

        """
                This function saves arrays to a .csv file
                :param name: (String) - the file name
                :param Titles: (String[]) - the titles of the columns
                :param Matrix: (float[][]) - the matrix of data
                :param withTitles: (boolean) - True if titles should be saved
                :return:
        """

        if len(Titles) != len(Matrix[0]):
            print("Columns don't match with titles!!")
        else:
            f = open(name + ".csv", 'w+')
            if withTitles:
                for i in range(len(Titles)):
                    if i < len(Titles) - 1:
                        f.write(Titles[i] + ',')
                    else:
                        f.write(Titles[i])
                f.write('\n')

            for i in range(len(Matrix)):
                for j in range(len(Matrix[i])):
                    if j < len(Matrix[i]) - 1:
                        f.write(str(Matrix[i][j]) + ',')
                    else:
                        f.write(str(Matrix[i][j]) + '\n')
            f.close()

    def open(self, file):

        """
        This function imports the geomtry and spring constants of the previously-deformed discretized fishing net
        :param file: the CSV file with the data
        :return: N/A
        """
        data = np.genfromtxt(file, delimiter=',', skip_header=1).T
        dimensions = np.array(data[26][:2], int)
        self.X, self.Y, self.Z = data[0].reshape(dimensions), data[1].reshape(dimensions), data[2].reshape(dimensions)
        self.Vx, self.Vy, self.Vz = data[3].reshape(dimensions), data[4].reshape(dimensions), data[5].reshape(dimensions)
        self.LN0, self.LS0, self.LE0, self.LW0 = data[6].reshape(dimensions), data[7].reshape(dimensions), data[8].reshape(dimensions), data[9].reshape(dimensions)
        self.LNE0, self.LNW0, self.LSE0, self.LSW0 = data[10].reshape(dimensions), data[11].reshape(dimensions), data[12].reshape(dimensions), data[13].reshape(dimensions)
        self.LNN0, self.LSS0, self.LEE0, self.LWW0 = data[14].reshape(dimensions), data[15].reshape(dimensions), data[16].reshape(dimensions), data[17].reshape(dimensions)
        self.kNE, self.kNW, self.kSE, self.kSW = data[18].reshape(dimensions), data[19].reshape(dimensions), data[20].reshape(dimensions), data[21].reshape(dimensions)
        self.kNN, self.kSS, self.kEE, self.kWW = data[22].reshape(dimensions), data[23].reshape(dimensions), data[24].reshape(dimensions), data[25].reshape(dimensions)
        self.dP = np.ones(self.X.shape) * self.dP[0][0]
        self.M = 0.5 * (self.LN0 + self.LS0) * 0.5 * (self.LE0 + self.LW0) * self.rho * np.pi / 4 * self.porosity
        self.Color = np.zeros(self.X.shape)
        self.Nx, self.Ny = self.X.shape
        self.computeNormals()

    def save(self, name):

        """
        This function saves the geometry and spring constants of the current fishing net in a CSV file
        :return: N/A
        """

        titles = ["X", "Y", "Z", "Vx", "Vy", "Vz", "LN0", "LS0", "LE0", "LW0", "LNE0", "LNW0", "LSE0", "LSW0", "LNN0", "LSS0", "LEE0", "LWW0", "kNE", "kNW", "kSE", "kSW", "kNN",
                  "kSS", "kEE", "kWW", "dimensions"]
        dimensions = np.zeros(len(self.X.flatten()))
        dimensions[0] = len(self.X)
        dimensions[1] = len(self.X[0])
        matrix = np.array(
            [self.X.flatten(), self.Y.flatten(), self.Z.flatten(), self.Vx.flatten(), self.Vy.flatten(), self.Vz.flatten(), self.LN0.flatten(), self.LS0.flatten(),
             self.LE0.flatten(), self.LW0.flatten(), self.LNE0.flatten(),
             self.LNW0.flatten(), self.LSE0.flatten(), self.LSW0.flatten(),
             self.LNN0.flatten(), self.LSS0.flatten(), self.LEE0.flatten(), self.LWW0.flatten(), self.kNE.flatten(), self.kNW.flatten(), self.kSE.flatten(),
             self.kSW.flatten(), self.kNN.flatten(), self.kSS.flatten(), self.kEE.flatten(), self.kWW.flatten(), dimensions])
        self.saveCSV(name, titles, matrix.T, True)

    def saveSTL(self):

        """
        This function saves an STL file of the current discretized fishing net, as well as STL files for all the individual panels used to simulate this net
        :return: N/A
        """

        X = self.X
        Y = self.Y
        Z = self.Z

        vertices = []
        indices = []

        for i in range(len(X)):
            for j in range(len(X[0])):
                vertices.append([X[i][j], Y[i][j], Z[i][j]])
                indices.append((i, j))

        vertices = np.array(vertices)
        faces = []

        for i in range(len(X) - 1):
            for j in range(len(X[0]) - 1):
                im = (i + 1) % len(X)
                jm = (j + 1) % len(X[0])
                t11 = i * len(X[0]) + j
                t12 = (i + 1) * len(X[0]) + j
                t13 = (i + 1) * len(X[0]) + j + 1
                t21 = i * len(X[0]) + j
                t22 = (i + 1) * len(X[0]) + j + 1
                t23 = i * len(X[0]) + j + 1
                faces.append([t11, t12, t13])
                faces.append([t21, t22, t23])

        faces = np.array(faces)

        net = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                net.vectors[i][j] = vertices[f[j], :]

        net.save('net.stl')

        for i in range(len(X) - 1):
            for j in range(len(X[0]) - 1):
                vertices = []
                vertices.append([X[i][j], Y[i][j], Z[i][j]])
                vertices.append([X[i][j + 1], Y[i][j + 1], Z[i][j + 1]])
                vertices.append([X[i + 1][j], Y[i + 1][j], Z[i + 1][j]])
                vertices.append([X[i + 1][j + 1], Y[i + 1][j + 1], Z[i + 1][j + 1]])
                vertices = np.array(vertices)
                faces = []
                faces.append([0, 1, 2])
                faces.append([1, 2, 3])
                faces = np.array(faces)
                panel = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for k, f in enumerate(faces):
                    for l in range(3):
                        panel.vectors[k][l] = vertices[f[l], :]
                panel.save('STL_Files/' + str(i * (len(X[0]) - 1) + j) + '.stl')


class Rope:
    """
    This class creates the mass-spring system used for discretizing the fishing net at an arbitrary resolution. It also simulates its deformation
    using the Euler time integration method
    """

    def __init__(self, point1, point2, resolution, material):

        """
        This constructor initializes the spring constants of the mass-spring system based on the E-modulus and geometry of the fishing net.
        It also initializes several other parameters of the water flow.
        :param X: the matrix containing the X-positions of the virtual nodes
        :param Y: the matrix containing the Y-positions of the virtual nodes
        :param Z: the matrix containing the Z-positions of the virtual nodes
        :param E_fibre: the E-modulus of the fibre used in twines (usually Nylon)
        :param twine_length: the length of one twine (distance between two knots)
        :param twine_density: the density of the material used in the twine
        :param knot_mass: the mass of the knots
        :param d_twine: the twine diameter
        :param Sn: the solidity ratio of the fishing net
        :param V_inf: the free-stream velocity of the fishing net
        :param rho: the density of the fluid in which the fishing net is deployed
        :param scale: the scaling factor of the fishing net geometry (used to speed up the simulation)
        """

        self.E_rope = material.E_fibre * np.pi / 4
        E_rope = self.E_rope
        self.DFiber = material.D_line
        self.X = np.linspace(point1[0], point2[0], resolution)
        self.Y = np.linspace(point1[1], point2[1], resolution)
        self.Z = np.linspace(point1[2], point2[2], resolution)
        self.Vx = np.zeros(self.X.shape)
        self.Vy = np.zeros(self.X.shape)
        self.Vz = np.zeros(self.X.shape)
        self.Nx = self.X.shape
        self.ECoeff = material.ElAx
        self.rho = material.rho
        XN, XS = np.roll(self.X, -1, 0), np.roll(self.X, 1, 0)
        YN, YS = np.roll(self.Y, -1, 0), np.roll(self.Y, 1, 0)
        ZN, ZS = np.roll(self.Z, -1, 0), np.roll(self.Z, 1, 0)
        XNN, XSS = np.roll(self.X, -2, 0), np.roll(self.X, 2, 0)
        YNN, YSS = np.roll(self.Y, -2, 0), np.roll(self.Y, 2, 0)
        ZNN, ZSS = np.roll(self.Z, -2, 0), np.roll(self.Z, 2, 0)

        dXN, dXS = XN - self.X, XS - self.X
        dYN, dYS = YN - self.Y, YS - self.Y
        dZN, dZS = ZN - self.Z, ZS - self.Z
        dXNN, dXSS = XNN - self.X, XSS - self.X
        dYNN, dYSS = YNN - self.Y, YSS - self.Y
        dZNN, dZSS = ZNN - self.Z, ZSS - self.Z

        self.LN0 = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
        self.LS0 = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
        self.LNN0 = (dXNN ** 2 + dYNN ** 2 + dZNN ** 2) ** 0.5
        self.LSS0 = (dXSS ** 2 + dYSS ** 2 + dZSS ** 2) ** 0.5

        self.kNN = E_rope * np.pi / 64 * material.D_line ** 4 / self.LNN0
        self.kSS = E_rope * np.pi / 64 * material.D_line ** 4 / self.LSS0
        self.kNN[-2:-1] = self.kSS[0:1] = 0

        self.M = 0.5 * (self.LN0 + self.LS0) * material.rho * np.pi / 4

    def getAcc(self, Fixed, dt):

        """
        This function computes the acceleration and performs the forward Euler time integration of every point in the discretized net
        :param X: the X-positions of the points in the net
        :param Y: the Y-positions of the points in the net
        :param Z: the Z-positions of the points in the net
        :param Vx: the X-velocities of the points in the net
        :param Vy: the Y-velocities of the points in the net
        :param Vz: the Z-velocities of the points in the net
        :param Fixed: the matrix of net points that should be fixed throughout the simulation
        :param dt: the time step
        :return: the new position and velocity arrays after one time step
        """
        X = self.X
        Y = self.Y
        Z = self.Z
        Vx = self.Vx
        Vy = self.Vy
        Vz = self.Vz
        g0 = 0
        gz = np.ones(self.M.shape) * g0
        C = 5 / self.X.shape[0]

        XN, XS = np.roll(self.X, -1, 0), np.roll(self.X, 1, 0)
        YN, YS = np.roll(self.Y, -1, 0), np.roll(self.Y, 1, 0)
        ZN, ZS = np.roll(self.Z, -1, 0), np.roll(self.Z, 1, 0)
        XNN, XSS = np.roll(self.X, -2, 0), np.roll(self.X, 2, 0)
        YNN, YSS = np.roll(self.Y, -2, 0), np.roll(self.Y, 2, 0)
        ZNN, ZSS = np.roll(self.Z, -2, 0), np.roll(self.Z, 2, 0)

        dXN, dXS = XN - X, XS - X
        dYN, dYS = YN - Y, YS - Y
        dZN, dZS = ZN - Z, ZS - Z
        dXNN, dXSS = XNN - X, XSS - X
        dYNN, dYSS = YNN - Y, YSS - Y
        dZNN, dZSS = ZNN - Z, ZSS - Z

        LN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
        LS = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
        LNN = (dXNN ** 2 + dYNN ** 2 + dZNN ** 2) ** 0.5
        LSS = (dXSS ** 2 + dYSS ** 2 + dZSS ** 2) ** 0.5

        FN = Fe(LN - self.LN0, self.ECoeff) / self.LN0
        FS = Fe(LS - self.LS0, self.ECoeff) / self.LS0

        LN = np.where(LN == 0, 1, LN)
        LS = np.where(LS == 0, 1, LS)
        LNN = np.where(LNN == 0, 1, LNN)
        LSS = np.where(LSS == 0, 1, LSS)

        FN[-1] = 0
        FS[0] = 0

        kNN = self.kNN
        kSS = self.kSS

        Ax = 1 / self.M * (FN * dXN / LN + FS * dXS / LS +
                           kNN * (LNN - self.LNN0) * dXNN / LNN + kSS * (LSS - self.LSS0) * dXSS / LSS
                           - C * Vx * Fixed)
        Ay = 1 / self.M * (FN * dYN / LN + FS * dYS / LS +
                           kNN * (LNN - self.LNN0) * dYNN / LNN + kSS * (LSS - self.LSS0) * dYSS / LSS
                           - C * Vy * Fixed)
        Az = 1 / self.M * (FN * dZN / LN + FS * dZS / LS +
                           kNN * (LNN - self.LNN0) * dZNN / LNN + kSS * (LSS - self.LSS0) * dZSS / LSS
                           - C * Vz * Fixed) + gz

        return Ax, Ay, Az

    def plot(self, ax, color):

        """
        This function plots the discretized fishing net in 3D for every 1000 time steps
        :param frame: the current time step
        :return: N/A
        """
        X = self.X
        Y = self.Y
        Z = self.Z
        ax.plot3D(X, Y, Z, color, linewidth=0.5)

    def saveCSV(self, name, Titles, Matrix, withTitles):

        """
                This function saves arrays to a .csv file
                :param name: (String) - the file name
                :param Titles: (String[]) - the titles of the columns
                :param Matrix: (float[][]) - the matrix of data
                :param withTitles: (boolean) - True if titles should be saved
                :return:
        """

        if len(Titles) != len(Matrix[0]):
            print("Columns don't match with titles!!")
        else:
            f = open(name + ".csv", 'w+')
            if withTitles:
                for i in range(len(Titles)):
                    if i < len(Titles) - 1:
                        f.write(Titles[i] + ',')
                    else:
                        f.write(Titles[i])
                f.write('\n')

            for i in range(len(Matrix)):
                for j in range(len(Matrix[i])):
                    if j < len(Matrix[i]) - 1:
                        f.write(str(Matrix[i][j]) + ',')
                    else:
                        f.write(str(Matrix[i][j]) + '\n')
            f.close()

    def open(self, file):

        """
        This function imports the geomtry and spring constants of the previously-deformed discretized fishing net
        :param file: the CSV file with the data
        :return: N/A
        """

        data = np.genfromtxt(file, delimiter=',', skip_header=1).T
        dimensions = np.array(data[12][0], int)
        self.X, self.Y, self.Z = data[0].reshape(dimensions), data[1].reshape(dimensions), data[2].reshape(dimensions)
        self.Vx, self.Vy, self.Vz = data[3].reshape(dimensions), data[4].reshape(dimensions), data[5].reshape(dimensions)
        self.LN0, self.LS0 = data[6].reshape(dimensions), data[7].reshape(dimensions)
        self.LNN0, self.LSS0 = data[8].reshape(dimensions), data[9].reshape(dimensions)
        self.kNN, self.kSS = data[10].reshape(dimensions), data[11].reshape(dimensions)
        self.Nx = self.X.shape
        self.M = 0.5 * (self.LN0 + self.LS0) * self.rho * np.pi / 4

    def save(self, name):

        """
        This function saves the geometry and spring constants of the current fishing net in a CSV file
        :return: N/A
        """

        titles = ["X", "Y", "Z", "Vx", "Vy", "Vz", "LN0", "LS0", "LNN0", "LSS0", "kNN", "kSS", "dimensions"]
        dimensions = np.zeros(len(self.X.flatten()))
        dimensions[0] = len(self.X)
        matrix = np.array([self.X.flatten(), self.Y.flatten(), self.Z.flatten(), self.Vx.flatten(), self.Vy.flatten(), self.Vz.flatten(), self.LN0.flatten(), self.LS0.flatten(),
                           self.LNN0.flatten(), self.LSS0.flatten(), self.kNN.flatten(), self.kSS.flatten(), dimensions])
        self.saveCSV(name, titles, matrix.T, True)

    def saveSTL(self):

        """
        This function saves an STL file of the current discretized fishing net, as well as STL files for all the individual panels used to simulate this net
        :return: N/A
        """

        X = self.X
        Y = self.Y
        Z = self.Z

        vertices = []
        indices = []

        for i in range(len(X)):
            for j in range(len(X[0])):
                vertices.append([X[i][j], Y[i][j], Z[i][j]])
                indices.append((i, j))

        vertices = np.array(vertices)
        faces = []

        for i in range(len(X) - 1):
            for j in range(len(X[0]) - 1):
                im = (i + 1) % len(X)
                jm = (j + 1) % len(X[0])
                t11 = i * len(X[0]) + j
                t12 = (i + 1) * len(X[0]) + j
                t13 = (i + 1) * len(X[0]) + j + 1
                t21 = i * len(X[0]) + j
                t22 = (i + 1) * len(X[0]) + j + 1
                t23 = i * len(X[0]) + j + 1
                faces.append([t11, t12, t13])
                faces.append([t21, t22, t23])

        faces = np.array(faces)

        net = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                net.vectors[i][j] = vertices[f[j], :]

        net.save('net.stl')

        for i in range(len(X) - 1):
            for j in range(len(X[0]) - 1):
                vertices = []
                vertices.append([X[i][j], Y[i][j], Z[i][j]])
                vertices.append([X[i][j + 1], Y[i][j + 1], Z[i][j + 1]])
                vertices.append([X[i + 1][j], Y[i + 1][j], Z[i + 1][j]])
                vertices.append([X[i + 1][j + 1], Y[i + 1][j + 1], Z[i + 1][j + 1]])
                vertices = np.array(vertices)
                faces = []
                faces.append([0, 1, 2])
                faces.append([1, 2, 3])
                faces = np.array(faces)
                panel = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for k, f in enumerate(faces):
                    for l in range(3):
                        panel.vectors[k][l] = vertices[f[l], :]
                panel.save('STL_Files/' + str(i * (len(X[0]) - 1) + j) + '.stl')


class Parachute:

    def __init__(self, disks, bands, num_suspension, num_gores, suspension_length, reinforcement, disk_res, band_res, sus_res, ang_res, canopy_material, sus_line_material,
                 reinforcement_material, dp_initial, closed_radius):
        self.Disks = []
        self.Bands = []
        self.SuspensionLines = []
        self.NumSus = num_suspension
        self.NumGores = num_gores
        self.CanopyMat = canopy_material
        self.SusMat = sus_line_material
        self.BandMat = canopy_material
        self.ReinfMat = reinforcement_material
        self.ReinfWidth = reinforcement
        self.dp = dp_initial
        self.rd = closed_radius

        ratio = 1

        for i in range(len(disks)):
            Nr = disk_res[i]
            Nang = ang_res
            r = np.linspace(disks[i][0], disks[i][1], Nr + 1)
            theta = np.linspace(0, np.pi * 2, Nang + 1)
            theta = theta[:-1].copy()
            R, Theta = np.meshgrid(r, theta)
            r_d = self.rd
            R_d = disks[i][1]
            R_v = disks[i][0]
            r_max = r_d / R_d * R + np.pi * R / (Nang + 1) * (1 - (r_d / R_d) ** 2) ** 0.5
            r_min = r_d / R_d * R - np.pi * R / (Nang + 1) * (1 - (r_d / R_d) ** 2) ** 0.5
            R_folded = r_max
            R_folded[1::2, :] = r_min[1::2, :]
            X = (R_folded * np.cos(Theta)).T
            Z = (R_folded * np.sin(Theta)).T
            Y = (-(R_d - R_v - R) * (1 - (r_d / R_d) ** 2) ** 0.5).T
            Disk = Canopy(X, Y, Z, canopy_material, reinforcement_material, sus_line_material, reinforcement, num_suspension, num_gores, dp_initial)
            self.Disks.append(Disk)
        for i in range(len(bands)):
            Nang = ang_res
            r = disks[-1][1]
            r_d = self.rd
            R_d = r
            r_max = r_d / R_d * r + np.pi * r / (Nang + 1) * (1 - (r_d / R_d) ** 2) ** 0.5
            r_min = r_d / R_d * r - np.pi * r / (Nang + 1) * (1 - (r_d / R_d) ** 2) ** 0.5
            theta = np.linspace(0, np.pi * 2, Nang + 1)
            theta = theta[:-1].copy()
            x = r_max * np.cos(theta)
            z = r_max * np.sin(theta)
            x[1::2] = (r_min * np.cos(theta))[1::2]
            z[1::2] = (r_min * np.sin(theta))[1::2]
            y = np.linspace(bands[i][0], bands[i][1], band_res[i])
            X, Y = np.meshgrid(x, y)
            Z, Y = np.meshgrid(z, y)
            Band = Canopy(X, Y, Z, canopy_material, reinforcement_material, sus_line_material, reinforcement, num_suspension, num_gores, dp_initial * ratio)
            self.Bands.append(Band)
        Sus_Index = np.linspace(0, ang_res, Num_Suspension + 1)
        self.Sus_Index = np.array(Sus_Index, int)[:-1]
        for i in range(len(self.Sus_Index)):
            index = self.Sus_Index[i]
            sus_line = []
            for j in range(len(disks) - 1):
                point1 = [self.Disks[j].X[-1, index], self.Disks[j].Y[-1, index], self.Disks[j].Z[-1, index]]
                point2 = [self.Disks[j + 1].X[0, index], self.Disks[j + 1].Y[0, index], self.Disks[j + 1].Z[0, index]]
                line = Rope(point1, point2, sus_res[j], sus_line_material)
                sus_line.append(line)
            if len(bands) > 0:
                point1 = [self.Disks[-1].X[-1, index], self.Disks[-1].Y[-1, index], self.Disks[-1].Z[-1, index]]
                point2 = [self.Bands[0].X[0, index], self.Bands[0].Y[0, index], self.Bands[0].Z[0, index]]
                line = Rope(point1, point2, sus_res[len(disks) - 1], sus_line_material)
                sus_line.append(line)
            for j in range(len(bands) - 1):
                point1 = [self.Bands[j].X[-1, index], self.Bands[j].Y[-1, index], self.Bands[j].Z[-1, index]]
                point2 = [self.Bands[j + 1].X[0, index], self.Bands[j + 1].Y[0, index], self.Bands[j + 1].Z[0, index]]
                line = Rope(point1, point2, sus_res[len(disks) + j], sus_line_material)
                sus_line.append(line)
            length = (suspension_length ** 2 - (self.rd) ** 2) ** 0.5
            self.Length = length
            if len(bands) > 0:
                point1 = [self.Bands[-1].X[-1, index], self.Bands[-1].Y[-1, index], self.Bands[-1].Z[-1, index]]
                point2 = [0, length, 0]
                line = Rope(point1, point2, sus_res[-1], sus_line_material)
                sus_line.append(line)
                self.SuspensionLines.append(sus_line)
            else:
                point1 = [self.Disks[-1].X[-1, index], self.Disks[-1].Y[-1, index], self.Disks[-1].Z[-1, index]]
                point2 = [0, length, 0]
                line = Rope(point1, point2, sus_res[-1], sus_line_material)
                sus_line.append(line)
                self.SuspensionLines.append(sus_line)

    def plot(self, frame):
        plt.figure(figsize=(6, 5))
        ax = plt.axes(projection='3d')
        scamap = plt.cm.ScalarMappable(cmap='jet')
        for disk in self.Disks:
            disk.plot(ax, 'blue', scamap)
        for band in self.Bands:
            band.plot(ax, 'blue', scamap)
        for sus_line in self.SuspensionLines:
            for line in sus_line:
                line.plot(ax, 'black')

        ax.set_zlim([-1, 1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1 + self.Length / 2, 1 + self.Length / 2])
        ax.view_init(elev=15., azim=20)

        plt.colorbar(scamap, orientation='vertical', label='Radial Stress [Mpa]', fraction=0.046, pad=0.04)
        plt.savefig('frames/' + str(frame) + '.png', dpi=400)
        # plt.show()
        plt.close()

    def plotDrag(self, frame, time, Drag):

        fig2 = plt.figure()
        ax = fig2.add_subplot()
        plt.suptitle(r"Drag Force vs. Time")

        ax.plot(time, Drag, linewidth=1, label='Parachute Drag Force')
        ax.set_xlabel(r"Time [s]")
        ax.set_ylabel(r"Drag [N]")
        # ax.set_ylim([0, 2.5])
        ax.legend()
        ax.grid()
        plt.savefig("DragLog/" + str(frame) + ".png", dpi=1000)
        plt.close(fig2)

    def computeDrag(self):

        Drag = 0

        for disk in self.Disks:
            X = disk.X
            Y = disk.Y
            Z = disk.Z

            XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
            YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
            ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)
            dXN, dXS, dXE, dXW = XN - X, XS - X, XE - X, XW - X
            dYN, dYS, dYE, dYW = YN - Y, YS - Y, YE - Y, YW - Y
            dZN, dZS, dZE, dZW = ZN - Z, ZS - Z, ZE - Z, ZW - Z

            LN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
            LS = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
            LE = (dXE ** 2 + dYE ** 2 + dZE ** 2) ** 0.5
            LW = (dXW ** 2 + dYW ** 2 + dZW ** 2) ** 0.5

            S = 0.5 * (LN + LS) * 0.5 * (LE + LW)
            S[-1, :] = S[-2, :]
            S[0, :] = S[1, :]
            Drag += np.sum(S * disk.nY * disk.dP)
        for band in self.Bands:
            X = band.X
            Y = band.Y
            Z = band.Z

            XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
            YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
            ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)
            dXN, dXS, dXE, dXW = XN - X, XS - X, XE - X, XW - X
            dYN, dYS, dYE, dYW = YN - Y, YS - Y, YE - Y, YW - Y
            dZN, dZS, dZE, dZW = ZN - Z, ZS - Z, ZE - Z, ZW - Z

            LN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
            LS = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
            LE = (dXE ** 2 + dYE ** 2 + dZE ** 2) ** 0.5
            LW = (dXW ** 2 + dYW ** 2 + dZW ** 2) ** 0.5

            S = 0.5 * (LN + LS) * 0.5 * (LE + LW)
            S[-1, :] = S[-2, :]
            S[0, :] = S[1, :]
            Drag += np.sum(S * band.nY * band.dP)

        return abs(Drag)

    def computeDragSus(self):
        Drag = 0

        for Susline in self.SuspensionLines:
            line = Susline[-1]
            dXN = line.X[-1] - line.X[-2]
            dYN = line.Y[-1] - line.Y[-2]
            dZN = line.Z[-1] - line.Z[-2]
            LN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
            dLN = LN - line.LN0[-2]
            Drag += Fe(np.where(LN - line.LN0[-2] < 0, 0, LN - line.LN0[-2]), line.ECoeff) / line.LN0[-2] * dYN / LN

        return Drag

    def computeNormals(self):
        nX = self.Disks[0].nX[0:-1].flatten()
        nY = self.Disks[0].nY[0:-1].flatten()
        nZ = self.Disks[0].nZ[0:-1].flatten()

        X = self.Disks[0].X
        Y = self.Disks[0].Y
        Z = self.Disks[0].Z

        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)

        XN2 = 0.5 * (XN + X)
        YN2 = 0.5 * (YN + Y)
        ZN2 = 0.5 * (ZN + Z)
        XS2 = 0.5 * (XS + X)
        YS2 = 0.5 * (YS + Y)
        ZS2 = 0.5 * (ZS + Z)
        XE2 = 0.5 * (XE + X)
        YE2 = 0.5 * (YE + Y)
        ZE2 = 0.5 * (ZE + Z)
        XW2 = 0.5 * (XW + X)
        YW2 = 0.5 * (YW + Y)
        ZW2 = 0.5 * (ZW + Z)

        dXNS = XN2 - XS2
        dYNS = YN2 - YS2
        dZNS = ZN2 - ZS2

        norm = (dXNS ** 2 + dYNS ** 2 + dZNS ** 2) ** 0.5

        dXNS = dXNS / norm
        dYNS = dYNS / norm
        dZNS = dZNS / norm

        dXNS = dXNS[0:-1].flatten()
        dYNS = dYNS[0:-1].flatten()
        dZNS = dZNS[0:-1].flatten()

        for i in range(1, len(self.Disks)):
            nX = np.concatenate((nX, self.Disks[i].nX[0:-1].flatten()), 0)
            nY = np.concatenate((nY, self.Disks[i].nY[0:-1].flatten()), 0)
            nZ = np.concatenate((nZ, self.Disks[i].nZ[0:-1].flatten()), 0)
            X = self.Disks[i].X
            Y = self.Disks[i].Y
            Z = self.Disks[i].Z

            XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
            YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
            ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)

            XN2 = 0.5 * (XN + X)
            YN2 = 0.5 * (YN + Y)
            ZN2 = 0.5 * (ZN + Z)
            XS2 = 0.5 * (XS + X)
            YS2 = 0.5 * (YS + Y)
            ZS2 = 0.5 * (ZS + Z)
            XE2 = 0.5 * (XE + X)
            YE2 = 0.5 * (YE + Y)
            ZE2 = 0.5 * (ZE + Z)
            XW2 = 0.5 * (XW + X)
            YW2 = 0.5 * (YW + Y)
            ZW2 = 0.5 * (ZW + Z)

            dXNSi = XN2 - XS2
            dYNSi = YN2 - YS2
            dZNSi = ZN2 - ZS2

            norm = (dXNSi ** 2 + dYNSi ** 2 + dZNSi ** 2) ** 0.5

            dXNSi = dXNSi / norm
            dYNSi = dYNSi / norm
            dZNSi = dZNSi / norm

            dXNS = np.concatenate((dXNS, dXNSi[0:-1].flatten()), 0)
            dYNS = np.concatenate((dYNS, dYNSi[0:-1].flatten()), 0)
            dZNS = np.concatenate((dZNS, dZNSi[0:-1].flatten()), 0)

        for i in range(len(self.Bands)):
            nX = np.concatenate((nX, self.Bands[i].nX[0:-1].flatten()), 0)
            nY = np.concatenate((nY, self.Bands[i].nY[0:-1].flatten()), 0)
            nZ = np.concatenate((nZ, self.Bands[i].nZ[0:-1].flatten()), 0)

            X = self.Bands[i].X
            Y = self.Bands[i].Y
            Z = self.Bands[i].Z

            XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = createNeighbours(X)
            YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = createNeighbours(Y)
            ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = createNeighbours(Z)

            XN2 = 0.5 * (XN + X)
            YN2 = 0.5 * (YN + Y)
            ZN2 = 0.5 * (ZN + Z)
            XS2 = 0.5 * (XS + X)
            YS2 = 0.5 * (YS + Y)
            ZS2 = 0.5 * (ZS + Z)
            XE2 = 0.5 * (XE + X)
            YE2 = 0.5 * (YE + Y)
            ZE2 = 0.5 * (ZE + Z)
            XW2 = 0.5 * (XW + X)
            YW2 = 0.5 * (YW + Y)
            ZW2 = 0.5 * (ZW + Z)

            dXNSi = XN2 - XS2
            dYNSi = YN2 - YS2
            dZNSi = ZN2 - ZS2

            norm = (dXNSi ** 2 + dYNSi ** 2 + dZNSi ** 2) ** 0.5

            dXNSi = dXNSi / norm
            dYNSi = dYNSi / norm
            dZNSi = dZNSi / norm

            dXNS = np.concatenate((dXNS, dXNSi[0:-1].flatten()), 0)
            dYNS = np.concatenate((dYNS, dYNSi[0:-1].flatten()), 0)
            dZNS = np.concatenate((dZNS, dZNSi[0:-1].flatten()), 0)

        return [nX, nY, nZ, dXNS, dYNS, dZNS]

    def stitch(self, part, line, index, orientation):
        C = 1
        if orientation == 1:
            dXN = line.X[1] - line.X[0]
            dYN = line.Y[1] - line.Y[0]
            dZN = line.Z[1] - line.Z[0]
            LN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
            LN0 = line.LN0[0]
            dXS = part.X[-2, index] - part.X[-1, index]
            dYS = part.Y[-2, index] - part.Y[-1, index]
            dZS = part.Z[-2, index] - part.Z[-1, index]
            LS = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
            LS0 = part.LS0[-1, index]
            dXE = part.X[-1, index + 1] - part.X[-1, index]
            dYE = part.Y[-1, index + 1] - part.Y[-1, index]
            dZE = part.Z[-1, index + 1] - part.Z[-1, index]
            LE = (dXE ** 2 + dYE ** 2 + dZE ** 2) ** 0.5
            LE0 = part.LE0[-1, index]
            dXW = part.X[-1, index - 1] - part.X[-1, index]
            dYW = part.Y[-1, index - 1] - part.Y[-1, index]
            dZW = part.Z[-1, index - 1] - part.Z[-1, index]
            LW = (dXW ** 2 + dYW ** 2 + dZW ** 2) ** 0.5
            LW0 = part.LW0[-1, index]
            dXSE = part.X[-2, index + 1] - part.X[-1, index]
            dYSE = part.Y[-2, index + 1] - part.Y[-1, index]
            dZSE = part.Z[-2, index + 1] - part.Z[-1, index]
            LSE = (dXSE ** 2 + dYSE ** 2 + dZSE ** 2) ** 0.5
            LSE0 = part.LSE0[-1, index]
            dXSW = part.X[-2, index - 1] - part.X[-1, index]
            dYSW = part.Y[-2, index - 1] - part.Y[-1, index]
            dZSW = part.Z[-2, index - 1] - part.Z[-1, index]
            LSW = (dXSW ** 2 + dYSW ** 2 + dZSW ** 2) ** 0.5
            LSW0 = part.LSE0[-1, index]
            FN = Fe(LN - LN0, line.ECoeff) / LN0
            FS = Fe(LS - LS0, part.ECoeffAx) / LS0 * 0.5 * (LE0 + LW0)
            FS += Fe(LS - LS0, part.ECoeffReinf) / LS0 * 0.5 * (LE0 + LW0)
            FE = Fe(LE - LE0, part.ECoeffAx) / LE0 * 0.5 * (LN0 + LS0)
            FE += Fe(LE - LE0, part.ECoeffReinf) / LE0 * 0.5 * (LN0 + LS0)
            FW = Fe(LW - LW0, part.ECoeffAx) / LW0 * 0.5 * (LN0 + LS0)
            FW += Fe(LW - LW0, part.ECoeffReinf) / LW0 * 0.5 * (LN0 + LS0)
            FSE = Fe(LSE - LSE0, part.ECoeffDiag) / LSE0 * 0.5 * (LSW0 + LSW0)
            FSE += Fe(LSE - LSE0, part.ECoeffReinfDiag) / LSE0 * 0.5 * (LSW0 + LSW0)
            FSW = Fe(LSW - LSW0, part.ECoeffDiag) / LSW0 * 0.5 * (LSE0 + LSE0)
            FSW += Fe(LSW - LSW0, part.ECoeffReinfDiag) / LSW0 * 0.5 * (LSE0 + LSE0)
            # if LN == 0: LN = 1
            # if LS == 0: LS = 1
            # if LE == 0: LE = 1
            # if LW == 0: LW = 1
            # if LSE == 0: LSE = 1
            # if LSW == 0: LSW = 1
            mass = part.M[-1, index] + line.M[0] + part.reinf_rho * 0.5 * (LE0 + LW0) * LS0 * np.pi / 4
            AxD = (FN * dXN / LN + FS * dXS / LS + FE * dXE / LE + FW * dXW / LW + FSE * dXSE / LSE + FSW * dXSW / LSW - C * part.Vx[-1, index]) / mass
            AyD = (FN * dYN / LN + FS * dYS / LS + FE * dYE / LE + FW * dYW / LW + FSE * dYSE / LSE + FSW * dYSW / LSW - C * part.Vy[-1, index]) / mass
            AzD = (FN * dZN / LN + FS * dZS / LS + FE * dZE / LE + FW * dZW / LW + FSE * dZSE / LSE + FSW * dZSW / LSW - C * part.Vz[-1, index]) / mass
            AxS = (FN * dXN / LN + FS * dXS / LS + FE * dXE / LE + FW * dXW / LW + FSE * dXSE / LSE + FSW * dXSW / LSW - C * line.Vx[0]) / mass
            AyS = (FN * dYN / LN + FS * dYS / LS + FE * dYE / LE + FW * dYW / LW + FSE * dYSE / LSE + FSW * dYSW / LSW - C * line.Vy[0]) / mass
            AzS = (FN * dZN / LN + FS * dZS / LS + FE * dZE / LE + FW * dZW / LW + FSE * dZSE / LSE + FSW * dZSW / LSW - C * line.Vz[0]) / mass
        else:
            dXN = part.X[1, index] - part.X[0, index]
            dYN = part.Y[1, index] - part.Y[0, index]
            dZN = part.Z[1, index] - part.Z[0, index]
            LN = (dXN ** 2 + dYN ** 2 + dZN ** 2) ** 0.5
            LN0 = part.LN0[0, index]
            dXS = line.X[-2] - line.X[-1]
            dYS = line.Y[-2] - line.Y[-1]
            dZS = line.Z[-2] - line.Z[-1]
            LS = (dXS ** 2 + dYS ** 2 + dZS ** 2) ** 0.5
            LS0 = line.LS0[-1]
            dXE = part.X[0, index + 1] - part.X[0, index]
            dYE = part.Y[0, index + 1] - part.Y[0, index]
            dZE = part.Z[0, index + 1] - part.Z[0, index]
            LE = (dXE ** 2 + dYE ** 2 + dZE ** 2) ** 0.5
            LE0 = part.LE0[0, index]
            dXW = part.X[0, index - 1] - part.X[0, index]
            dYW = part.Y[0, index - 1] - part.Y[0, index]
            dZW = part.Z[0, index - 1] - part.Z[0, index]
            LW = (dXW ** 2 + dYW ** 2 + dZW ** 2) ** 0.5
            LW0 = part.LW0[0, index]
            dXNE = part.X[1, index + 1] - part.X[0, index]
            dYNE = part.Y[1, index + 1] - part.Y[0, index]
            dZNE = part.Z[1, index + 1] - part.Z[0, index]
            LNE = (dXNE ** 2 + dYNE ** 2 + dZNE ** 2) ** 0.5
            LNE0 = part.LNE0[0, index]
            dXNW = part.X[1, index - 1] - part.X[0, index]
            dYNW = part.Y[1, index - 1] - part.Y[0, index]
            dZNW = part.Z[1, index - 1] - part.Z[0, index]
            LNW = (dXNW ** 2 + dYNW ** 2 + dZNW ** 2) ** 0.5
            LNW0 = part.LNW0[0, index]
            FN = Fe(LN - LN0, part.ECoeffAx) / LN0 * 0.5 * (LE0 + LW0)
            FN += Fe(LN - LN0, part.ECoeffReinf) / LN0 * 0.5 * (LE0 + LW0)
            FS = Fe(LS - LS0, line.ECoeff) / LS0
            FE = Fe(LE - LE0, part.ECoeffAx) / LE0 * 0.5 * (LN0 + LS0)
            FE += Fe(LE - LE0, part.ECoeffReinf) / LE0 * 0.5 * (LN0 + LS0)
            FW = Fe(LW - LW0, part.ECoeffAx) / LW0 * 0.5 * (LN0 + LS0)
            FW += Fe(LW - LW0, part.ECoeffReinf) / LW0 * 0.5 * (LN0 + LS0)
            FNE = Fe(LNE - LNE0, part.ECoeffDiag) / LNE0 * 0.5 * (LNW0 + LNW0)
            FNE += Fe(LNE - LNE0, part.ECoeffReinfDiag) / LNE0 * 0.5 * (LNW0 + LNW0)
            FNW = Fe(LNW - LNW0, part.ECoeffDiag) / LNW0 * 0.5 * (LNE0 + LNE0)
            FNW += Fe(LNW - LNW0, part.ECoeffReinfDiag) / LNW0 * 0.5 * (LNE0 + LNE0)
            # if LN == 0: LN = 1
            # if LS == 0: LS = 1
            # if LE == 0: LE = 1
            # if LW == 0: LW = 1
            # if LNE == 0: LNE = 1
            # if LNW == 0: LNW = 1
            mass = part.M[0, index] + line.M[-1] + part.reinf_rho * 0.5 * (LE0 + LW0) * LN0 * np.pi / 4
            AxD = (FN * dXN / LN + FS * dXS / LS + FE * dXE / LE + FW * dXW / LW + FNE * dXNE / LNE + FNW * dXNW / LNW - C * part.Vx[0, index]) / mass
            AyD = (FN * dYN / LN + FS * dYS / LS + FE * dYE / LE + FW * dYW / LW + FNE * dYNE / LNE + FNW * dYNW / LNW - C * part.Vy[0, index]) / mass
            AzD = (FN * dZN / LN + FS * dZS / LS + FE * dZE / LE + FW * dZW / LW + FNE * dZNE / LNE + FNW * dZNW / LNW - C * part.Vz[0, index]) / mass
            AxS = (FN * dXN / LN + FS * dXS / LS + FE * dXE / LE + FW * dXW / LW + FNE * dXNE / LNE + FNW * dXNW / LNW - C * part.Vx[0, index]) / mass
            AyS = (FN * dYN / LN + FS * dYS / LS + FE * dYE / LE + FW * dYW / LW + FNE * dYNE / LNE + FNW * dYNW / LNW - C * part.Vy[0, index]) / mass
            AzS = (FN * dZN / LN + FS * dZS / LS + FE * dZE / LE + FW * dZW / LW + FNE * dZNE / LNE + FNW * dZNW / LNW - C * part.Vz[0, index]) / mass

        return AxD, AyD, AzD, AxS, AyS, AzS

    def solveStep(self, dt):

        AccDisks = []
        for i in range(len(self.Disks)):
            Fixed = np.ones(self.Disks[i].X.shape)
            Ax, Ay, Az = self.Disks[i].getAcc(Fixed, dt)
            AccDisks.append([Ax, Ay, Az])
        AccBands = []
        for i in range(len(self.Bands)):
            Fixed = np.ones(self.Bands[i].X.shape)
            Ax, Ay, Az = self.Bands[i].getAcc(Fixed, dt)
            AccBands.append([Ax, Ay, Az])
        AccSusLines = []
        for i in range(len(self.SuspensionLines)):
            AccLines = []
            for j in range(len(self.SuspensionLines[i])):
                line = self.SuspensionLines[i][j]
                Fixed = np.ones(line.X.shape)
                Ax, Ay, Az = line.getAcc(Fixed, dt)
                AccLines.append([Ax, Ay, Az])
            AccSusLines.append(AccLines)
        for i in range(len(self.Disks)):
            for j in range(len(self.SuspensionLines)):
                index = self.Sus_Index[j]
                disk = self.Disks[i]
                line = self.SuspensionLines[j][i]
                AxD, AyD, AzD, AxS, AyS, AzS = self.stitch(disk, line, index, 1)
                AccDisks[i][0][-1, index] = AxD
                AccDisks[i][1][-1, index] = AyD
                AccDisks[i][2][-1, index] = AzD
                AccSusLines[j][i][0][0] = AxS
                AccSusLines[j][i][1][0] = AyS
                AccSusLines[j][i][2][0] = AzS
                if i < len(self.Disks) - 1:
                    disk2 = self.Disks[i + 1]
                    AxD, AyD, AzD, AxS, AyS, AzS = self.stitch(disk2, line, index, -1)
                    AccDisks[i + 1][0][0, index] = AxD
                    AccDisks[i + 1][1][0, index] = AyD
                    AccDisks[i + 1][2][0, index] = AzD
                    AccSusLines[j][i][0][-1] = AxS
                    AccSusLines[j][i][1][-1] = AyS
                    AccSusLines[j][i][2][-1] = AzS
                # print(AccSusLines[j][0][1])

        for i in range(len(self.Bands)):
            offset = len(self.Disks)
            for j in range(len(self.SuspensionLines)):
                index = self.Sus_Index[j]
                band = self.Bands[i]
                line = self.SuspensionLines[j][i + offset]
                AxD, AyD, AzD, AxS, AyS, AzS = self.stitch(band, line, index, 1)
                AccBands[i][0][-1, index] = AxD
                AccBands[i][1][-1, index] = AyD
                AccBands[i][2][-1, index] = AzD
                AccSusLines[j][i + offset][0][0] = AxS
                AccSusLines[j][i + offset][1][0] = AyS
                AccSusLines[j][i + offset][2][0] = AzS
                if i < len(self.Bands) - 1:
                    band2 = self.Bands[i + 1]
                    AxD, AyD, AzD, AxS, AyS, AzS = self.stitch(band2, line, index, -1)
                    AccBands[i + 1][0][0, index] = AxD
                    AccBands[i + 1][1][0, index] = AyD
                    AccBands[i + 1][2][0, index] = AzD
                    AccSusLines[j][i + offset][0][-1] = AxS
                    AccSusLines[j][i + offset][1][-1] = AyS
                    AccSusLines[j][i + offset][2][-1] = AzS
        offset = len(self.Disks)
        if len(self.Bands) > 0:
            for j in range(len(self.SuspensionLines)):
                index = self.Sus_Index[j]
                band = self.Bands[0]
                line = self.SuspensionLines[j][offset - 1]
                AxD, AyD, AzD, AxS, AyS, AzS = self.stitch(band, line, index, -1)
                AccBands[0][0][0, index] = AxD
                AccBands[0][1][0, index] = AyD
                AccBands[0][2][0, index] = AzD
                AccSusLines[j][offset - 1][0][-1] = AxS
                AccSusLines[j][offset - 1][1][-1] = AyS
                AccSusLines[j][offset - 1][2][-1] = AzS
                # print(AccSusLines[j][offset-1][1])

        for i in range(len(self.Disks)):
            self.Disks[i].Vx += AccDisks[i][0] * dt
            self.Disks[i].Vy += AccDisks[i][1] * dt
            self.Disks[i].Vz += AccDisks[i][2] * dt
            self.Disks[i].X += self.Disks[i].Vx * dt
            self.Disks[i].Y += self.Disks[i].Vy * dt
            self.Disks[i].Z += self.Disks[i].Vz * dt

        for i in range(len(self.Bands)):
            self.Bands[i].Vx += AccBands[i][0] * dt
            self.Bands[i].Vy += AccBands[i][1] * dt
            self.Bands[i].Vz += AccBands[i][2] * dt
            self.Bands[i].X += self.Bands[i].Vx * dt
            self.Bands[i].Y += self.Bands[i].Vy * dt
            self.Bands[i].Z += self.Bands[i].Vz * dt

        for i in range(len(self.SuspensionLines)):
            for j in range(len(self.SuspensionLines[i]) - 1):
                self.SuspensionLines[i][j].Vx += AccSusLines[i][j][0] * dt
                self.SuspensionLines[i][j].Vy += AccSusLines[i][j][1] * dt
                self.SuspensionLines[i][j].Vz += AccSusLines[i][j][2] * dt
                self.SuspensionLines[i][j].X += self.SuspensionLines[i][j].Vx * dt
                self.SuspensionLines[i][j].Y += self.SuspensionLines[i][j].Vy * dt
                self.SuspensionLines[i][j].Z += self.SuspensionLines[i][j].Vz * dt
            self.SuspensionLines[i][-1].Vx[:-1] += (AccSusLines[i][-1][0] * dt)[:-1]
            self.SuspensionLines[i][-1].Vy[:-1] += (AccSusLines[i][-1][1] * dt)[:-1]
            self.SuspensionLines[i][-1].Vz[:-1] += (AccSusLines[i][-1][2] * dt)[:-1]
            self.SuspensionLines[i][-1].X[:-1] += (self.SuspensionLines[i][-1].Vx * dt)[:-1]
            self.SuspensionLines[i][-1].Y[:-1] += (self.SuspensionLines[i][-1].Vy * dt)[:-1]
            self.SuspensionLines[i][-1].Z[:-1] += (self.SuspensionLines[i][-1].Vz * dt)[:-1]

    def solver(self, T, Nt, frames):
        dt = T / Nt
        step = int(Nt / frames)
        time = []
        Drag = []
        for i in range(Nt):
            self.solveStep(dt)
            if i % 10 == 0: print("Computing iteration", int(i))
            if i % step == 0:
                time.append(i * dt)
                Drag.append(self.computeDrag())
                self.plotDrag(int(i / step), time, Drag)
                self.plot(int(i / step))
                self.saveSTL('SPEARIIChuteScaled_Closed')

            # if i % (3 * step):
            #     self.saveParachute('SPEARIIChuteScaled')

    def saveSTL(self, file):

        vertices = []
        indices = []
        faces = []
        offset = 0
        for disk in self.Disks:
            X = disk.X
            Y = disk.Y
            Z = disk.Z
            XU = X + disk.nX * disk.DFiber / 2
            YU = Y + disk.nY * disk.DFiber / 2
            ZU = Z + disk.nZ * disk.DFiber / 2
            XL = X - disk.nX * disk.DFiber / 2
            YL = Y - disk.nY * disk.DFiber / 2
            ZL = Z - disk.nZ * disk.DFiber / 2

            for i in range(len(X)):
                for j in range(len(X[0])):
                    vertices.append([XU[i][j], YU[i][j], ZU[i][j]])
                    indices.append((i, j))
            for i in range(len(X)):
                for j in range(len(X[0])):
                    vertices.append([XL[i][j], YL[i][j], ZL[i][j]])
                    indices.append((i, j))

            for i in range(len(X) - 1):
                for j in range(len(X[0])):
                    jp = (j + 1) % len(X[0])
                    t11U = i * len(X[0]) + j + offset
                    t12U = (i + 1) * len(X[0]) + j + offset
                    t13U = (i + 1) * len(X[0]) + jp + offset
                    t21U = i * len(X[0]) + j + offset
                    t22U = (i + 1) * len(X[0]) + jp + offset
                    t23U = i * len(X[0]) + jp + offset
                    faces.append([t11U, t12U, t13U])
                    faces.append([t21U, t22U, t23U])
                    t11L = i * len(X[0]) + j + offset + X.shape[0] * X.shape[1]
                    t12L = (i + 1) * len(X[0]) + j + offset + X.shape[0] * X.shape[1]
                    t13L = (i + 1) * len(X[0]) + jp + offset + X.shape[0] * X.shape[1]
                    t21L = i * len(X[0]) + j + offset + X.shape[0] * X.shape[1]
                    t22L = (i + 1) * len(X[0]) + jp + offset + X.shape[0] * X.shape[1]
                    t23L = i * len(X[0]) + jp + offset + X.shape[0] * X.shape[1]
                    faces.append([t11L, t12L, t13L])
                    faces.append([t21L, t22L, t23L])
            for j in range(X.shape[1]):
                jp = (j + 1) % len(X[0])
                t11U = j + offset
                t12U = jp + offset
                t13U = jp + offset + X.shape[0] * X.shape[1]
                t21U = j + offset
                t22U = j + offset + X.shape[0] * X.shape[1]
                t23U = jp + offset + X.shape[0] * X.shape[1]
                faces.append([t11U, t12U, t13U])
                faces.append([t21U, t22U, t23U])
                t11L = j + offset + (X.shape[0] - 1) * X.shape[1]
                t12L = jp + offset + (X.shape[0] - 1) * X.shape[1]
                t13L = jp + offset + X.shape[0] * X.shape[1] + (X.shape[0] - 1) * X.shape[1]
                t21L = j + offset + (X.shape[0] - 1) * X.shape[1]
                t22L = j + offset + X.shape[0] * X.shape[1] + (X.shape[0] - 1) * X.shape[1]
                t23L = jp + offset + X.shape[0] * X.shape[1] + (X.shape[0] - 1) * X.shape[1]
                faces.append([t11L, t12L, t13L])
                faces.append([t21L, t22L, t23L])

            offset += X.shape[0] * X.shape[1] * 2
        for band in self.Bands:
            X = band.X
            Y = band.Y
            Z = band.Z
            XU = X + band.nX * band.DFiber / 2
            YU = Y + band.nY * band.DFiber / 2
            ZU = Z + band.nZ * band.DFiber / 2
            XL = X - band.nX * band.DFiber / 2
            YL = Y - band.nY * band.DFiber / 2
            ZL = Z - band.nZ * band.DFiber / 2

            for i in range(len(X)):
                for j in range(len(X[0])):
                    vertices.append([XU[i][j], YU[i][j], ZU[i][j]])
                    indices.append((i, j))
            for i in range(len(X)):
                for j in range(len(X[0])):
                    vertices.append([XL[i][j], YL[i][j], ZL[i][j]])
                    indices.append((i, j))

            for i in range(len(X) - 1):
                for j in range(len(X[0])):
                    jp = (j + 1) % len(X[0])
                    t11U = i * len(X[0]) + j + offset
                    t12U = (i + 1) * len(X[0]) + j + offset
                    t13U = (i + 1) * len(X[0]) + jp + offset
                    t21U = i * len(X[0]) + j + offset
                    t22U = (i + 1) * len(X[0]) + jp + offset
                    t23U = i * len(X[0]) + jp + offset
                    faces.append([t11U, t12U, t13U])
                    faces.append([t21U, t22U, t23U])
                    t11L = i * len(X[0]) + j + offset + X.shape[0] * X.shape[1]
                    t12L = (i + 1) * len(X[0]) + j + offset + X.shape[0] * X.shape[1]
                    t13L = (i + 1) * len(X[0]) + jp + offset + X.shape[0] * X.shape[1]
                    t21L = i * len(X[0]) + j + offset + X.shape[0] * X.shape[1]
                    t22L = (i + 1) * len(X[0]) + jp + offset + X.shape[0] * X.shape[1]
                    t23L = i * len(X[0]) + jp + offset + X.shape[0] * X.shape[1]
                    faces.append([t11L, t12L, t13L])
                    faces.append([t21L, t22L, t23L])
            for j in range(X.shape[1]):
                jp = (j + 1) % len(X[0])
                t11U = j + offset
                t12U = jp + offset
                t13U = jp + offset + X.shape[0] * X.shape[1]
                t21U = j + offset
                t22U = j + offset + X.shape[0] * X.shape[1]
                t23U = jp + offset + X.shape[0] * X.shape[1]
                faces.append([t11U, t12U, t13U])
                faces.append([t21U, t22U, t23U])
                t11L = j + offset + (X.shape[0] - 1) * X.shape[1]
                t12L = jp + offset + (X.shape[0] - 1) * X.shape[1]
                t13L = jp + offset + X.shape[0] * X.shape[1] + (X.shape[0] - 1) * X.shape[1]
                t21L = j + offset + (X.shape[0] - 1) * X.shape[1]
                t22L = j + offset + X.shape[0] * X.shape[1] + (X.shape[0] - 1) * X.shape[1]
                t23L = jp + offset + X.shape[0] * X.shape[1] + (X.shape[0] - 1) * X.shape[1]
                faces.append([t11L, t12L, t13L])
                faces.append([t21L, t22L, t23L])
            offset += X.shape[0] * X.shape[1] * 2
        vertices = np.array(vertices)
        faces = np.array(faces)

        object = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                object.vectors[i][j] = vertices[f[j], :]

        object.save(file + '.stl')

    def saveSTLs(self, file):

        if not os.path.exists(file):
            os.makedirs(file)
        index = 0
        for disk in self.Disks:
            X = disk.X
            Y = disk.Y
            Z = disk.Z
            for i in range(0, len(X) - 1):
                for j in range(len(X[0])):
                    vertices = []
                    jp = (j + 1) % len(X[0])
                    vertices.append([X[i][j], Y[i][j], Z[i][j]])
                    vertices.append([X[i][jp], Y[i][jp], Z[i][jp]])
                    vertices.append([X[i + 1][j], Y[i + 1][j], Z[i + 1][j]])
                    vertices.append([X[i + 1][jp], Y[i + 1][jp], Z[i + 1][jp]])
                    vertices = np.array(vertices)
                    faces = []
                    faces.append([0, 1, 2])
                    faces.append([1, 2, 3])
                    faces = np.array(faces)
                    panel = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                    for k, f in enumerate(faces):
                        for l in range(3):
                            panel.vectors[k][l] = vertices[f[l], :]
                    panel.save(file + '/panel' + str(index) + '.stl')
                    index += 1

        for band in self.Bands:
            X = band.X
            Y = band.Y
            Z = band.Z
            for i in range(0, len(X) - 1):
                for j in range(len(X[0])):
                    vertices = []
                    jp = (j + 1) % len(X[0])
                    vertices.append([X[i][j], Y[i][j], Z[i][j]])
                    vertices.append([X[i][jp], Y[i][jp], Z[i][jp]])
                    vertices.append([X[i + 1][j], Y[i + 1][j], Z[i + 1][j]])
                    vertices.append([X[i + 1][jp], Y[i + 1][jp], Z[i + 1][jp]])
                    vertices = np.array(vertices)
                    faces = []
                    faces.append([0, 1, 2])
                    faces.append([1, 2, 3])
                    faces = np.array(faces)
                    panel = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                    for k, f in enumerate(faces):
                        for l in range(3):
                            panel.vectors[k][l] = vertices[f[l], :]
                    panel.save(file + '/panel' + str(index) + '.stl')
                    index += 1

    def saveParachute(self, name):

        if not os.path.exists(name):
            os.makedirs(name)
        iter = 0
        for disk in self.Disks:
            disk.save(name + '/Disk' + str(iter))
            iter += 1
        iter = 0
        for band in self.Bands:
            band.save(name + '/Band' + str(iter))
            iter += 1
        for i in range(len(self.SuspensionLines)):
            for j in range(len(self.SuspensionLines[0])):
                line = self.SuspensionLines[i][j]
                line.save(name + '/Line' + str(i) + '_' + str(j))

    def importParachute(self, name):

        files = os.listdir(name)

        diskNames = [i for i in files if 'Disk' in i]
        bandNames = [i for i in files if 'Band' in i]
        lineNames = [i for i in files if 'Line' in i]
        lineNumbers = []
        for lineName in lineNames:
            lineNumbers.append(int(lineName.replace('_', '')[4:-4]))
        lineIndices = np.array(lineNumbers).argsort()
        lineNames = np.array(lineNames)[lineIndices]
        lineNames = lineNames.reshape((self.NumSus, int(len(lineNames) / (self.NumSus))))
        # lineNames = np.flip(lineNames, 0)

        self.Disks = []
        self.Bands = []
        self.SuspensionLines = []

        for diskName in diskNames:
            X = np.zeros((4, 4))
            Y = np.zeros((4, 4))
            Z = np.zeros((4, 4))
            disk = Canopy(X, Y, Z, self.CanopyMat, self.ReinfMat, self.SusMat, self.ReinfWidth, self.NumSus, self.NumGores, self.dp)
            disk.open(name + '/' + diskName)
            self.Disks.append(disk)
        for bandName in bandNames:
            X = np.zeros((4, 4))
            Y = np.zeros((4, 4))
            Z = np.zeros((4, 4))
            band = Canopy(X, Y, Z, self.CanopyMat, self.ReinfMat, self.SusMat, self.ReinfWidth, self.NumSus, self.NumGores, self.dp)
            band.open(name + '/' + bandName)
            self.Bands.append(band)
        for i in range(len(lineNames)):
            Line = []
            for j in range(len(lineNames[i])):
                line = Rope([0, 0, 0], [1, 1, 1], 10, self.SusMat)
                line.open(name + '/' + lineNames[i][j])
                Line.append(line)
            self.SuspensionLines.append(Line)

        for i in range(len(self.SuspensionLines)):
            for j in range(len(self.Disks)):
                self.SuspensionLines[i][j].X[0] = self.Disks[j].X[-1, self.Sus_Index[i]]
                self.SuspensionLines[i][j].Y[0] = self.Disks[j].Y[-1, self.Sus_Index[i]]
                self.SuspensionLines[i][j].Z[0] = self.Disks[j].Z[-1, self.Sus_Index[i]]
                if j < len(self.Disks) - 1:
                    self.SuspensionLines[i][j].X[-1] = self.Disks[j + 1].X[0, self.Sus_Index[i]]
                    self.SuspensionLines[i][j].Y[-1] = self.Disks[j + 1].Y[0, self.Sus_Index[i]]
                    self.SuspensionLines[i][j].Z[-1] = self.Disks[j + 1].Z[0, self.Sus_Index[i]]

        offset = len(self.Disks)

        for i in range(len(self.SuspensionLines)):
            for j in range(len(self.Bands)):
                self.SuspensionLines[i][j + offset].X[0] = self.Bands[j].X[-1, self.Sus_Index[i]]
                self.SuspensionLines[i][j + offset].Y[0] = self.Bands[j].Y[-1, self.Sus_Index[i]]
                self.SuspensionLines[i][j + offset].Z[0] = self.Bands[j].Z[-1, self.Sus_Index[i]]
                if j < len(self.Bands) - 1:
                    self.SuspensionLines[i][j + offset].X[-1] = self.Bands[j + 1].X[0, self.Sus_Index[i]]
                    self.SuspensionLines[i][j + offset].Y[-1] = self.Bands[j + 1].Y[0, self.Sus_Index[i]]
                    self.SuspensionLines[i][j + offset].Z[-1] = self.Bands[j + 1].Z[0, self.Sus_Index[i]]

    def importP(self, outputPfile, resolution, bounds):

        """
        This function imports the velocity and pressure data from the OpenFOAM simulation of the previously-deformed fishing net geometry
        :param outputUfile: the OpenFOAM log file containing the velocity and pressure fields surrounding the net
        :param netfile: the CSV file containing the previous geometry of the fishing net
        :param resolution: the resolution of the computational domain used in OpenFOAM
        :param bounds: the bounds of the computational domain used in OpenFOAM
        :return: N/A
        """

        data = np.genfromtxt(outputPfile, skip_header=23, skip_footer=19)
        P = data.reshape((resolution[2], resolution[1], resolution[0]))

        x = np.linspace(bounds[0][0], bounds[0][1], resolution[0])
        y = np.linspace(bounds[1][0], bounds[1][1], resolution[1])
        z = np.linspace(bounds[2][0], bounds[2][1], resolution[2])

        # P = np.flip(P, 2)

        plt.imshow(P[50, :, :])
        plt.colorbar(orientation='horizontal')
        plt.show()

        p = interpolate.RegularGridInterpolator((x, y, z), P, 'linear')
        for disk in self.Disks:
            d = np.max([disk.DFiber / 2, np.max([(bounds[0][1] - bounds[0][0]) / resolution[0],
                                                 (bounds[1][1] - bounds[1][0]) / resolution[1],
                                                 (bounds[2][1] - bounds[2][0]) / resolution[2]])])
            Pu = p((disk.X + d * disk.nX, disk.Y + d * disk.nY, disk.Z + d * disk.nZ))
            Pl = p((disk.X - d * disk.nX, disk.Y - d * disk.nY, disk.Z - d * disk.nZ))
            dP = Pu - Pl
            disk.dP = -dP
        for band in self.Bands:
            d = np.max([band.DFiber / 2, np.max([(bounds[0][1] - bounds[0][0]) / resolution[0],
                                                 (bounds[1][1] - bounds[1][0]) / resolution[1],
                                                 (bounds[2][1] - bounds[2][0]) / resolution[2]])])
            Pu = p((band.X + d * band.nX, band.Y + d * band.nY, band.Z + d * band.nZ))
            Pl = p((band.X - d * band.nX, band.Y - d * band.nY, band.Z - d * band.nZ))
            dP = Pu - Pl
            band.dP = -dP


RipNylon = Canopy_Material(2.7e6, 0.048, [15781 * 2, 506452 * 2], [100], 0.5e-3, 0.95)
Aramid = Canopy_Material(3.4e6, 0.03, [21520, 150645.2], [0], 0.5e-3, 1)
Spectra = Suspension_Material(1.1e7, 0.0027, [5380.3, -967333.5, 167075000], 4e-3)

Disks = [[0.072313 / 2, 1.038776 / 2]]
# Disks = [[0.1,  0.3], [0.4,  0.6], [0.7,  0.9], [1,  1.2], [1.3,  1.5]]
Bands = [[0.12111, 0.12111 + 0.170766]]
# Bands = []
Suspension_Length = 1.5
Num_Suspension = 6
Num_Gores = 6
Reinforcement_Width = 12.5e-3
# Disk_Resolution = [120]
Disk_Resolution = [90]
Band_Resolution = [30]
Angular_Resolution = 216
Sus_Resolution = [3, 15]

initial_pressure = 970

T = 0.01
Nt = 15000
frames = 50

parachute = Parachute(Disks, Bands, Num_Suspension, Num_Gores, Suspension_Length, Reinforcement_Width, Disk_Resolution, Band_Resolution, Sus_Resolution, Angular_Resolution,
                      RipNylon,
                      Spectra, Aramid, initial_pressure, 1.038776 / 2)
parachute.importParachute('SPEARIIChuteScaled_Closed')
parachute.importP('p', [100, 200, 100], [[-1.5, 1.5], [-3, 2], [-1.5, 1.5]])
parachute.solver(T, Nt, frames)
parachute.saveParachute('SPEARIIChuteScaled_Closed')
