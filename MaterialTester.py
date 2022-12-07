import numpy as np
from matplotlib import pylab as plt

import Parachute3D as par


class Sample:
    """
    This class creates the mass-spring system used for discretizing the fishing net at an arbitrary resolution. It also simulates its deformation
    using the Euler time integration method
    """

    def __init__(self, X, Y, Z, canopy_material):

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
        self.Nx, self.Ny = X.shape
        self.ECoeffAx = np.array(canopy_material.ElAx)
        self.ECoeffDiag = np.array(canopy_material.ElDiag)
        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = par.createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = par.createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = par.createNeighbours(Z)

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
        self.Fixed = np.ones(X.shape)
        self.Fixed[0, :] = 0
        self.Fixed[-1, :] = 0

        self.M = 0.5 * (self.LN0 + self.LS0) * 0.5 * (self.LE0 + self.LW0) * canopy_material.rho * np.pi / 4 * self.porosity

    def pull(self, Force):
        ForceX = np.zeros(self.X.shape)
        ForceY = np.zeros(self.X.shape)
        ForceZ = np.zeros(self.X.shape)

        ld = (self.LE0 + self.LW0)[-1, int(len(self.LE0[0]) / 2)]

        ForceY[-1, :] = Force * ld / 2 / 0.1

        return ForceX, ForceY, ForceZ, 0, 0, 0

    def pullDisp(self, dx):
        maxStrain = 1.5
        dL = ((self.X[-1] - self.X[-2])**2 + (self.Y[-1] - self.Y[-2])**2 + (self.Z[-1] - self.Z[-2])**2)**0.5
        strain = np.average(dL / self.LS0[-1])
        if strain < maxStrain: self.Y[-1, :] += dx

    def computeTensileForce(self, type):
        X = self.X
        Y = self.Y
        Z = self.Z
        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = par.createNeighbours(self.X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = par.createNeighbours(self.Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = par.createNeighbours(self.Z)

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

        if (type == 'axial'):

            FN = par.Fe(LN / self.LN0 - 1, self.ECoeffAx) * 0.5 * (self.LE0 + self.LW0)
            FS = par.Fe(LS / self.LS0 - 1, self.ECoeffAx) * 0.5 * (self.LE0 + self.LW0)
            FE = par.Fe(LE / self.LE0 - 1, self.ECoeffAx) * 0.5 * (self.LN0 + self.LS0)
            FW = par.Fe(LW / self.LW0 - 1, self.ECoeffAx) * 0.5 * (self.LN0 + self.LS0)

            FNE = par.Fe(LNE / self.LNE0 - 1, self.ECoeffDiag) * 0.5 * (self.LNW0 + self.LSE0)
            FNW = par.Fe(LNW / self.LNW0 - 1, self.ECoeffDiag) * 0.5 * (self.LNE0 + self.LSW0)
            FSE = par.Fe(LSE / self.LSE0 - 1, self.ECoeffDiag) * 0.5 * (self.LNE0 + self.LSW0)
            FSW = par.Fe(LSW / self.LSW0 - 1, self.ECoeffDiag) * 0.5 * (self.LNW0 + self.LSE0)
            FNN = par.Fe((LNN / self.LNN0 - 1), self.ECoeffAx) * ((self.LE0 + self.LW0) / self.LNN0) / 10000
            FSS = par.Fe((LSS / self.LSS0 - 1), self.ECoeffAx) * ((self.LE0 + self.LW0) / self.LSS0) / 10000
            FEE = par.Fe((LEE / self.LEE0 - 1), self.ECoeffAx) * ((self.LN0 + self.LS0) / self.LEE0) / 10000
            FWW = par.Fe((LWW / self.LWW0 - 1), self.ECoeffAx) * ((self.LN0 + self.LS0) / self.LWW0) / 10000

        else:
            FN = par.Fe(LN / self.LN0 - 1, self.ECoeffDiag) * 0.5 * (self.LE0 + self.LW0)
            FS = par.Fe(LS / self.LS0 - 1, self.ECoeffDiag) * 0.5 * (self.LE0 + self.LW0)
            FE = par.Fe(LE / self.LE0 - 1, self.ECoeffDiag) * 0.5 * (self.LN0 + self.LS0)
            FW = par.Fe(LW / self.LW0 - 1, self.ECoeffDiag) * 0.5 * (self.LN0 + self.LS0)

            FNE = par.Fe(LNE / self.LNE0 - 1, self.ECoeffAx / 2) * 0.5 * (self.LNW0 + self.LSE0)
            FNW = par.Fe(LNW / self.LNW0 - 1, self.ECoeffAx / 2) * 0.5 * (self.LNE0 + self.LSW0)
            FSE = par.Fe(LSE / self.LSE0 - 1, self.ECoeffAx / 2) * 0.5 * (self.LNE0 + self.LSW0)
            FSW = par.Fe(LSW / self.LSW0 - 1, self.ECoeffAx / 2) * 0.5 * (self.LNW0 + self.LSE0)
            FNN = par.Fe((LNN / self.LNN0 - 1), self.ECoeffAx) * ((self.LE0 + self.LW0)) / 1000000
            FSS = par.Fe((LSS / self.LSS0 - 1), self.ECoeffAx) * ((self.LE0 + self.LW0)) / 1000000
            FEE = par.Fe((LEE / self.LEE0 - 1), self.ECoeffAx) * ((self.LN0 + self.LS0)) / 1000000
            FWW = par.Fe((LWW / self.LWW0 - 1), self.ECoeffAx) * ((self.LN0 + self.LS0)) / 1000000

        Force = np.sum(
            ((FN * dYN / LN +
              FNE * dYNE / LNE + FNW * dYNW / LNW + FNN * dYNN / LNN))[-4, 1:-1]
        )
        return Force

    def getAcc(self, dt, velocity, Force, type):

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

        self.pullDisp(velocity * dt)

        X = self.X
        Y = self.Y
        Z = self.Z
        Vx = self.Vx
        Vy = self.Vy
        Vz = self.Vz
        Fixed = self.Fixed
        g0 = 0
        gz = np.ones(self.M.shape) * g0
        C = 1 / X.shape[0] / X.shape[1]
        CGl = 4 / X.shape[0] / X.shape[1]

        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = par.createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = par.createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = par.createNeighbours(Z)

        VXN, VXS, VXE, VXW, VXNE, VXNW, VXSE, VXSW, VXNN, VXSS, VXEE, VXWW = par.createNeighbours(Vx)
        VYN, VYS, VYE, VYW, VYNE, VYNW, VYSE, VYSW, VYNN, VYSS, VYEE, VYWW = par.createNeighbours(Vy)
        VZN, VZS, VZE, VZW, VZNE, VZNW, VZSE, VZSW, VZNN, VZSS, VZEE, VZWW = par.createNeighbours(Vz)

        dXN, dXS, dXE, dXW = XN - X, XS - X, XE - X, XW - X
        dYN, dYS, dYE, dYW = YN - Y, YS - Y, YE - Y, YW - Y
        dZN, dZS, dZE, dZW = ZN - Z, ZS - Z, ZE - Z, ZW - Z
        dXNE, dXNW, dXSE, dXSW = XNE - X, XNW - X, XSE - X, XSW - X
        dYNE, dYNW, dYSE, dYSW = YNE - Y, YNW - Y, YSE - Y, YSW - Y
        dZNE, dZNW, dZSE, dZSW = ZNE - Z, ZNW - Z, ZSE - Z, ZSW - Z
        dXNN, dXSS, dXEE, dXWW = XNN - X, XSS - X, XEE - X, XWW - X
        dYNN, dYSS, dYEE, dYWW = YNN - Y, YSS - Y, YEE - Y, YWW - Y
        dZNN, dZSS, dZEE, dZWW = ZNN - Z, ZSS - Z, ZEE - Z, ZWW - Z

        dVXN, dVXS, dVXE, dVXW = VXN - Vx, VXS - Vx, VXE - Vx, VXW - Vx
        dVYN, dVYS, dVYE, dVYW = VYN - Vy, VYS - Vy, VYE - Vy, VYW - Vy
        dVZN, dVZS, dVZE, dVZW = VZN - Vz, VZS - Vz, VZE - Vz, VZW - Vz
        dVXNE, dVXNW, dVXSE, dVXSW = VXNE - Vx, VXNW - Vx, VXSE - Vx, VXSW - Vx
        dVYNE, dVYNW, dVYSE, dVYSW = VYNE - Vy, VYNW - Vy, VYSE - Vy, VYSW - Vy
        dVZNE, dVZNW, dVZSE, dVZSW = VZNE - Vz, VZNW - Vz, VZSE - Vz, VZSW - Vz
        dVXNN, dVXSS, dVXEE, dVXWW = VXNN - Vx, VXSS - Vx, VXEE - Vx, VXWW - Vx
        dVYNN, dVYSS, dVYEE, dVYWW = VYNN - Vy, VYSS - Vy, VYEE - Vy, VYWW - Vy
        dVZNN, dVZSS, dVZEE, dVZWW = VZNN - Vz, VZSS - Vz, VZEE - Vz, VZWW - Vz

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

        dVN = dVXN * dXN / LN + dVYN * dYN / LN + dVZN * dZN / LN
        dVS = dVXS * dXS / LS + dVYS * dYS / LS + dVZS * dZS / LS
        dVE = dVXE * dXE / LE + dVYE * dYE / LE + dVZE * dZE / LE
        dVW = dVXW * dXW / LW + dVYW * dYW / LW + dVZW * dZW / LW
        dVNE = dVXNE * dXNE / LNE + dVYNE * dYNE / LNE + dVZNE * dZNE / LNE
        dVSE = dVXSE * dXSE / LSE + dVYSE * dYSE / LSE + dVZSE * dZSE / LSE
        dVNW = dVXNW * dXNW / LNW + dVYNW * dYNW / LNW + dVZNW * dZNW / LNW
        dVSW = dVXSW * dXSW / LSW + dVYSW * dYSW / LSW + dVZSW * dZSW / LSW
        dVNN = dVXNN * dXNN / LNN + dVYNN * dYNN / LNN + dVZNN * dZNN / LNN
        dVSS = dVXSS * dXSS / LSS + dVYSS * dYSS / LSS + dVZSS * dZSS / LSS
        dVEE = dVXEE * dXEE / LEE + dVYEE * dYEE / LEE + dVZEE * dZEE / LEE
        dVWW = dVXWW * dXWW / LWW + dVYWW * dYWW / LWW + dVZWW * dZWW / LWW

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

        if (type == 'axial'):

            FN = par.Fe(LN / self.LN0 - 1, self.ECoeffAx) * 0.5 * (self.LE0 + self.LW0)
            FS = par.Fe(LS / self.LS0 - 1, self.ECoeffAx) * 0.5 * (self.LE0 + self.LW0)
            FE = par.Fe(LE / self.LE0 - 1, self.ECoeffAx) * 0.5 * (self.LN0 + self.LS0)
            FW = par.Fe(LW / self.LW0 - 1, self.ECoeffAx) * 0.5 * (self.LN0 + self.LS0)

            FNE = par.Fe(LNE / self.LNE0 - 1, self.ECoeffDiag) * 0.5 * (self.LNW0 + self.LSE0)
            FNW = par.Fe(LNW / self.LNW0 - 1, self.ECoeffDiag) * 0.5 * (self.LNE0 + self.LSW0)
            FSE = par.Fe(LSE / self.LSE0 - 1, self.ECoeffDiag) * 0.5 * (self.LNE0 + self.LSW0)
            FSW = par.Fe(LSW / self.LSW0 - 1, self.ECoeffDiag) * 0.5 * (self.LNW0 + self.LSE0)
            FNN = par.Fe((LNN / self.LNN0 - 1), self.ECoeffAx) * ((self.LE0 + self.LW0)) / 1000000
            FSS = par.Fe((LSS / self.LSS0 - 1), self.ECoeffAx) * ((self.LE0 + self.LW0)) / 1000000
            FEE = par.Fe((LEE / self.LEE0 - 1), self.ECoeffAx) * ((self.LN0 + self.LS0)) / 1000000
            FWW = par.Fe((LWW / self.LWW0 - 1), self.ECoeffAx) * ((self.LN0 + self.LS0)) / 1000000

        else:
            FN = par.Fe(LN / self.LN0 - 1, self.ECoeffDiag) * 0.5 * (self.LE0 + self.LW0)
            FS = par.Fe(LS / self.LS0 - 1, self.ECoeffDiag) * 0.5 * (self.LE0 + self.LW0)
            FE = par.Fe(LE / self.LE0 - 1, self.ECoeffDiag) * 0.5 * (self.LN0 + self.LS0)
            FW = par.Fe(LW / self.LW0 - 1, self.ECoeffDiag) * 0.5 * (self.LN0 + self.LS0)

            FNE = par.Fe(LNE / self.LNE0 - 1, self.ECoeffAx / 2) * 0.5 * (self.LNW0 + self.LSE0)
            FNW = par.Fe(LNW / self.LNW0 - 1, self.ECoeffAx / 2) * 0.5 * (self.LNE0 + self.LSW0)
            FSE = par.Fe(LSE / self.LSE0 - 1, self.ECoeffAx / 2) * 0.5 * (self.LNE0 + self.LSW0)
            FSW = par.Fe(LSW / self.LSW0 - 1, self.ECoeffAx / 2) * 0.5 * (self.LNW0 + self.LSE0)
            FNN = par.Fe((LNN / self.LNN0 - 1), self.ECoeffAx) * ((self.LE0 + self.LW0)) * 0
            FSS = par.Fe((LSS / self.LSS0 - 1), self.ECoeffAx) * ((self.LE0 + self.LW0)) * 0
            FEE = par.Fe((LEE / self.LEE0 - 1), self.ECoeffAx) * ((self.LN0 + self.LS0)) * 0
            FWW = par.Fe((LWW / self.LWW0 - 1), self.ECoeffAx) * ((self.LN0 + self.LS0)) * 0

        FN[-1, :] = 0
        FNE[-1, :] = 0
        FNW[-1, :] = 0
        FS[0, :] = 0
        FSE[0, :] = 0
        FSW[0, :] = 0
        FE[:, 0] = 0
        FNE[:, 0] = 0
        FSE[:, 0] = 0
        FW[:, -1] = 0
        FNW[:, -1] = 0
        FSW[:, -1] = 0
        FNN[-1, :] = 0
        FNN[-2, :] = 0
        FSS[0, :] = 0
        FSS[1, :] = 0
        FEE[:, 0] = 0
        FEE[:, 1] = 0
        FWW[:, -1] = 0
        FWW[:, -2] = 0

        Dx, Dy, Dz, Lx, Ly, Lz = self.pull(0)

        Ax = 1 / self.M * (FN * dXN / LN + FS * dXS / LS + FE * dXE / LE + FW * dXW / LW +
                           FNE * dXNE / LNE + FNW * dXNW / LNW +
                           FSE * dXSE / LSE + FSW * dXSW / LSW +
                           FNN * dXNN / LNN + FSS * dXSS / LSS +
                           FEE * dXEE / LEE + FWW * dXWW / LWW +
                           C * (dVN * dXN / LN + dVS * dXS / LS +
                                dVE * dXE / LE + dVW * dXW / LW +
                                dVNE * dXNE / LNE + dVNW * dXNW / LNW +
                                dVSE * dXSE / LSE + dVSW * dXSW / LSW +
                                dVNN * dXNN / LNN + dVSS * dXSS / LSS +
                                dVEE * dXEE / LEE + dVWW * dXWW / LWW) * Fixed - CGl * Vx * Fixed + Dx + Lx)
        Ay = 1 / self.M * (FN * dYN / LN + FS * dYS / LS + FE * dYE / LE + FW * dYW / LW +
                           FNE * dYNE / LNE + FNW * dYNW / LNW +
                           FSE * dYSE / LSE + FSW * dYSW / LSW +
                           FNN * dYNN / LNN + FSS * dYSS / LSS +
                           FEE * dYEE / LEE + FWW * dYWW / LWW +
                           C * (dVN * dYN / LN + dVS * dYS / LS +
                                dVE * dYE / LE + dVW * dYW / LW +
                                dVNE * dYNE / LNE + dVNW * dYNW / LNW +
                                dVSE * dYSE / LSE + dVSW * dYSW / LSW +
                                dVNN * dYNN / LNN + dVSS * dYSS / LSS +
                                dVEE * dYEE / LEE + dVWW * dYWW / LWW) * Fixed - CGl * Vy * Fixed + Dy + Ly)
        Az = 1 / self.M * (FN * dZN / LN + FS * dZS / LS + FE * dZE / LE + FW * dZW / LW +
                           FNE * dZNE / LNE + FNW * dZNW / LNW +
                           FSE * dZSE / LSE + FSW * dZSW / LSW +
                           FNN * dZNN / LNN + FSS * dZSS / LSS +
                           FEE * dZEE / LEE + FWW * dZWW / LWW +
                           C * (dVN * dZN / LN + dVS * dZS / LS +
                                dVE * dZE / LE + dVW * dZW / LW +
                                dVNE * dZNE / LNE + dVNW * dZNW / LNW +
                                dVSE * dZSE / LSE + dVSW * dZSW / LSW +
                                dVNN * dZNN / LNN + dVSS * dZSS / LSS +
                                dVEE * dZEE / LEE + dVWW * dZWW / LWW) * Fixed - CGl * Vz * Fixed + Dz + Lz) + gz

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
        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = par.createNeighbours(self.X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = par.createNeighbours(self.Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = par.createNeighbours(self.Z)

        dXN, dXS, dXE, dXW = XN - X, XS - X, XE - X, XW - X
        dYN, dYS, dYE, dYW = YN - Y, YS - Y, YE - Y, YW - Y
        dZN, dZS, dZE, dZW = ZN - Z, ZS - Z, ZE - Z, ZW - Z
        for i in range(len(X)):
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
        FeNS = par.Fe(StrainNS, self.ECoeffAx) / 0.5 / (self.LN0 + self.LS0)
        FeEW = par.Fe(StrainEW, self.ECoeffAx) / 0.5 / (self.LE0 + self.LW0)
        FeTot = par.Fe(Strain, self.ECoeffAx)
        StressNS = FeNS / self.DFiber
        StressNS[0, :] = StressNS[1, :].copy()
        StressNS[-1, :] = StressNS[-2, :].copy()
        StressEW = FeEW / self.DFiber
        StressEW[0, :] = StressEW[1, :].copy()
        StressEW[-1, :] = StressEW[-2, :].copy()

        StressVM = ((StressEW - StressNS) ** 2 * 0.5) ** 0.5

        fcolors = scamap.to_rgba(StressVM / 10 ** 6)
        surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, lw=0.6, antialiased=True, vmin=np.min(StressVM), vmax=np.max(StressVM))

        # for i in range(0, len(X), 2):
        #     for j in range(0, len(X[0]), 2):
        #         ax.plot3D([X[i][j], X[i][j] + self.nX[i][j]*0.1], [Y[i][j], Y[i][j] + self.nY[i][j]*0.1], [Z[i][j], Z[i][j] + self.nZ[i][j]*0.1], 'red', linewidth = 0.4)
        ax.view_init(elev=90., azim=90)

k = 0.2
RipNylon = par.Canopy_Material(2.7e6, 0.048, [23202 * (1 - k), 49578 * (1 - k), -153205 * (1 - k)], [23202 * k / 2, 49578 * k / 2, -153205 * k / 2], 0.09e-3, 1)
RipNylon = par.Canopy_Material(2.7e6, 0.048, [30006 * (1 - k), -18936 * (1 - k)], [ 30006 * k / 2, -18936 * k / 2], 0.09e-3, 1)
#RipNylon = par.Canopy_Material(2.7e6, 0.048, [1300], [5022], 0.09e-3, 1)

Nx = 75
Ny = 150
Nt = 500000
T = 0.1

Lx, Rx = -0.0445, 0.0445
Ly, Ry = -0.0565, 0.0565

x = np.linspace(Lx, Rx, Nx)
y = np.linspace(Ly, Ry, Ny)

X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

Force = 1000
velocity = 6

sample = Sample(X, Y, Z, RipNylon)

time = []
displacement = []
TensileForce = []

data_exp = np.genfromtxt('MaterialTesting/CSVData/CanopyData.csv', delimiter=',').T
print(data_exp)
strain_ax_exp = data_exp[0]
force_ax_exp = data_exp[1]
strain_sh_exp = data_exp[2]
force_sh_exp = data_exp[3]
L0_ax_exp = 0.113
width_ax_exp = 0.089
L0_sh_exp = 0.114
width_sh_exp = 0.091

for i in range(Nt):
    dt = T / Nt
    Ax, Ay, Az = sample.getAcc(dt, velocity, Force, 'axial')
    Fixed = sample.Fixed
    sample.Vx += Ax * dt * Fixed
    sample.Vy += Ay * dt * Fixed
    sample.Vz += Az * dt * Fixed
    sample.X += sample.Vx * dt * Fixed
    sample.Y += sample.Vy * dt * Fixed
    sample.Z += sample.Vz * dt * Fixed

    if i % 1000 == 0:
        print("Computing iteration", i)
        plt.rcParams.update({'font.size': 6})
        plt.figure(figsize=(6, 5))
        ax = plt.axes(projection='3d')
        scamap = plt.cm.ScalarMappable(cmap='jet')
        ax.set_zlim([-0.1, 0.1])
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        sample.plot(ax, 'blue', scamap)
        plt.colorbar(scamap, orientation='vertical', label='Von-Mises Stress [MPa]', fraction=0.046, pad=0.04)
        plt.savefig('MaterialTesting/frames/' + str(int(i / 1000)) + '.png', dpi=500)
        plt.close()

        time.append(i * dt)
        displacement.append(i * dt * velocity * 1000)
        TensileForce.append(sample.computeTensileForce('axial'))

        plt.figure(figsize=(6, 5))
        plt.rcParams.update({'font.size': 10})
        plt.plot(displacement, TensileForce, color='black', linewidth=2, label='Simulated Elastic Response')
        plt.plot(strain_ax_exp * L0_ax_exp * 1000, force_ax_exp * width_ax_exp, color='red', linestyle='--', linewidth=1, label='Experimental Elastic Response')
        plt.grid()
        plt.legend()
        plt.xlabel("Displacement [mm]")
        plt.ylabel("Force [N]")
        plt.suptitle("Force-Displacement Curve")
        plt.savefig('MaterialTesting/ForceGraph/' + str(int(i / 1000)) + '.png', dpi=300)
        plt.close()
