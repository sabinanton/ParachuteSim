import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np
import Parachute3D
from scipy import interpolate



def plotParachute(parachute):

    factor = 1

    for disk in parachute.Disks:
        X = disk.X
        Y = disk.Y
        Z = disk.Z

        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = Parachute3D.createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = Parachute3D.createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = Parachute3D.createNeighbours(Z)

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

        Fe = Parachute3D.Fe

        FN = Fe(LN / disk.LN0 - 1, disk.ECoeffAx) * 0.5 * (disk.LE0 + disk.LW0)
        FS = Fe(LS / disk.LS0 - 1, disk.ECoeffAx) * 0.5 * (disk.LE0 + disk.LW0)
        FE = Fe(LE / disk.LE0 - 1, disk.ECoeffAx) * 0.5 * (disk.LN0 + disk.LS0)
        FW = Fe(LW / disk.LW0 - 1, disk.ECoeffAx) * 0.5 * (disk.LN0 + disk.LS0)

        FNE = Fe(LNE / disk.LNE0 - 1, disk.ECoeffDiag) * 0.5 * (disk.LNW0 + disk.LSE0)
        FNW = Fe(LNW / disk.LNW0 - 1, disk.ECoeffDiag) * 0.5 * (disk.LNE0 + disk.LSW0)
        FSE = Fe(LSE / disk.LSE0 - 1, disk.ECoeffDiag) * 0.5 * (disk.LNE0 + disk.LSW0)
        FSW = Fe(LSW / disk.LSW0 - 1, disk.ECoeffDiag) * 0.5 * (disk.LNW0 + disk.LSE0)
        FNN = 0  # Fe((LNN / disk.LNN0) - 1, disk.ECoeffAx) * ((disk.LE0 + disk.LW0)) / 100
        FSS = 0  # Fe((LSS / disk.LSS0) - 1, disk.ECoeffAx) * ((disk.LE0 + disk.LW0)) / 100
        FEE = 0  # Fe((LEE / disk.LEE0) - 1, disk.ECoeffAx) * ((disk.LN0 + disk.LS0)) / 100
        FWW = 0  # Fe((LWW / disk.LWW0) - 1, disk.ECoeffAx) * ((disk.LN0 + disk.LS0)) / 100

        thickness = np.ones(X.shape) * disk.DFiber
        thickness[:disk.RNumWidthNS, :] += disk.DFiber_Reinf
        thickness[-disk.RNumWidthNS:, :] += disk.DFiber_Reinf
        for i in disk.Gore_Index:
            im = min(i - 1, int(i - disk.RNumWidthEW / 2))  # % len(FE[0])
            ip = max(i + 1, int(i + disk.RNumWidthEW / 2))  # % len(FE[0])
            if im < 0 and ip > 0:
                thickness[:, im:] += disk.DFiber_Reinf
                thickness[:, :ip] += disk.DFiber_Reinf

            else:
                thickness[:, im:ip] += disk.DFiber_Reinf


        StressNS = (0.5 * (FN + FS + FNN + FSS) + (FNE + FSE + FNW + FSW) * 0.5 / 2**0.5) / 0.5 / (disk.LE0 + disk.LW0) / thickness
        StressEW = (0.5 * (FE + FW + FEE + FWW) + (FNE + FSE + FNW + FSW) * 0.5 / 2**0.5) / 0.5 / (disk.LN0 + disk.LS0) / thickness
        StressNS[0, :] = StressNS[1, :].copy()
        StressNS[-1, :] = StressNS[-2, :].copy()
        StressEW[0, :] = StressEW[1, :].copy()
        StressEW[-1, :] = StressEW[-2, :].copy()

        StressVM = ((StressEW - StressNS) ** 2 * 0.5) ** 0.5
        maxStressVM = np.max(StressVM)
        minStressVM = np.min(StressVM)

        StressVM = np.where(StressVM < minStressVM / factor, minStressVM / factor, StressVM)
        StressVM = np.where(StressVM > maxStressVM * factor, maxStressVM * factor, StressVM)

        Dmesh = mlab.mesh(X, Y, Z, scalars=StressVM / 10**6, colormap='jet')
        Dmesh.actor.property.interpolation = 'phong'
        Dmesh.actor.property.specular = 0.1
        Dmesh.actor.property.specular_power = 5

    for band in parachute.Bands:
        X = band.X
        Y = band.Y
        Z = band.Z

        XN, XS, XE, XW, XNE, XNW, XSE, XSW, XNN, XSS, XEE, XWW = Parachute3D.createNeighbours(X)
        YN, YS, YE, YW, YNE, YNW, YSE, YSW, YNN, YSS, YEE, YWW = Parachute3D.createNeighbours(Y)
        ZN, ZS, ZE, ZW, ZNE, ZNW, ZSE, ZSW, ZNN, ZSS, ZEE, ZWW = Parachute3D.createNeighbours(Z)

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

        Fe = Parachute3D.Fe

        FN = Fe(LN / band.LN0 - 1, band.ECoeffAx) * 0.5 * (band.LE0 + band.LW0)
        FS = Fe(LS / band.LS0 - 1, band.ECoeffAx) * 0.5 * (band.LE0 + band.LW0)
        FE = Fe(LE / band.LE0 - 1, band.ECoeffAx) * 0.5 * (band.LN0 + band.LS0)
        FW = Fe(LW / band.LW0 - 1, band.ECoeffAx) * 0.5 * (band.LN0 + band.LS0)

        FNE = Fe(LNE / band.LNE0 - 1, band.ECoeffDiag) * 0.5 * (band.LNW0 + band.LSE0)
        FNW = Fe(LNW / band.LNW0 - 1, band.ECoeffDiag) * 0.5 * (band.LNE0 + band.LSW0)
        FSE = Fe(LSE / band.LSE0 - 1, band.ECoeffDiag) * 0.5 * (band.LNE0 + band.LSW0)
        FSW = Fe(LSW / band.LSW0 - 1, band.ECoeffDiag) * 0.5 * (band.LNW0 + band.LSE0)
        FNN = 0  # Fe((LNN / band.LNN0) - 1, band.ECoeffAx) * ((band.LE0 + band.LW0)) / 100
        FSS = 0  # Fe((LSS / band.LSS0) - 1, band.ECoeffAx) * ((band.LE0 + band.LW0)) / 100
        FEE = 0  # Fe((LEE / band.LEE0) - 1, band.ECoeffAx) * ((band.LN0 + band.LS0)) / 100
        FWW = 0  # Fe((LWW / band.LWW0) - 1, band.ECoeffAx) * ((band.LN0 + band.LS0)) / 100

        thickness = np.ones(X.shape) * band.DFiber
        thickness[:band.RNumWidthNS, :] += band.DFiber_Reinf
        thickness[-band.RNumWidthNS:, :] += band.DFiber_Reinf
        for i in band.Gore_Index:
            im = min(i - 1, int(i - band.RNumWidthEW / 2))  # % len(FE[0])
            ip = max(i + 1, int(i + band.RNumWidthEW / 2))  # % len(FE[0])
            if im < 0 and ip > 0:
                thickness[:, im:] += band.DFiber_Reinf
                thickness[:, :ip] += band.DFiber_Reinf

            else:
                thickness[:, im:ip] += band.DFiber_Reinf

        StressNS = (0.5 * (FN + FS + FNN + FSS) + (FNE + FSE + FNW + FSW) * 0.5 / 2 ** 0.5) / 0.5 / (band.LE0 + band.LW0) / thickness
        StressEW = (0.5 * (FE + FW + FEE + FWW) + (FNE + FSE + FNW + FSW) * 0.5 / 2 ** 0.5) / 0.5 / (band.LN0 + band.LS0) / thickness
        StressNS[0, :] = StressNS[1, :].copy()
        StressNS[-1, :] = StressNS[-2, :].copy()
        StressEW[0, :] = StressEW[1, :].copy()
        StressEW[-1, :] = StressEW[-2, :].copy()

        StressVM = ((StressEW - StressNS) ** 2 * 0.5) ** 0.5
        maxStressVM = np.max(StressVM)
        minStressVM = np.min(StressVM)
        StressVM = np.where(StressVM < minStressVM / factor, minStressVM / factor, StressVM)
        StressVM = np.where(StressVM > maxStressVM * factor, maxStressVM * factor, StressVM)

        Bmesh = mlab.mesh(X, Y, Z, scalars=StressVM / 10**6, colormap='jet')
        Bmesh.actor.property.interpolation = 'phong'
        Bmesh.actor.property.specular = 0.1
        Bmesh.actor.property.specular_power = 5
    for sus_line in parachute.SuspensionLines:
        for line in sus_line:
            l = mlab.plot3d(line.X, line.Y, line.Z, tube_radius=0.0025)

    mlab.show()


if __name__ == '__main__':
    ########### Non-linear Materials ###################

    k = 0.2
    RipNylon = Parachute3D.Canopy_Material(2.7e6, 0.048, [16615 * (1 - k), 67021 * (1 - k)], [16615 *  k / 2, 67021 *  k / 2], 0.09e-3, 1)
    Aramid = Parachute3D.Canopy_Material(3.4e6, 0.03, [211985, 489633], [500], 0.43e-3, 1)
    Spectra = Parachute3D.Suspension_Material(1.1e7, 0.0027, [3961.9, -130540, 6e6], 4e-3)

    # ########### Linear Materials ###################
    #
    # k = 0.1
    # RipNylon = Canopy_Material(2.7e6, 0.048, [16615 * (1 - k)], [16615 *  k / 2], 0.09e-3, 1)
    # Aramid = Canopy_Material(3.4e6, 0.03, [211985], [500], 0.43e-3, 1)
    # Spectra = Suspension_Material(1.1e7, 0.0027, [3961.9], 4e-3)

    ################# Stratos IV Parachute ###################

    Disks = [[0.1 / 2, 1.0036 / 2]]
    Bands = [[0.062, 0.062 + 0.293]]


    ################# Walrus Parachute ##################

    # Disks = [[0.17 / 2, 1.3 / 2]]
    # Bands = [[0.085, 0.085 + 0.4]]

    Suspension_Length = 1.328
    Num_Suspension = 18
    Num_Gores = 6
    Reinforcement_Width = 25e-3
    # Disk_Resolution = [120]
    Disk_Resolution = [60]
    Band_Resolution = [30]
    Angular_Resolution = 216
    Sus_Resolution = [5, 15]

    parachute = Parachute3D.Parachute(Disks, Bands, Num_Suspension, Num_Gores, Suspension_Length, Reinforcement_Width, Disk_Resolution, Band_Resolution, Sus_Resolution, Angular_Resolution,
                          RipNylon,
                          Spectra, Aramid, 500, 1.0036 / 2)
    parachute.importParachute('StratosIVNonLinear')

    plotParachute(parachute)

