import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from scipy import interpolate
from stl import mesh
import OpenFOAM


def create_panel(p1, p2, N, k, kp, kn, X, Y, kN, kS, S, Solid, D, end):
    dx = (p2[0] - p1[0]) / N
    dy = (p2[1] - p1[1]) / N
    D = interpolate.interp1d([0, N], D, 'linear')
    kS.append(kp(D(0)))
    S.append(0)
    for i in range(N):
        X.append(p1[0] + i * dx)
        Y.append(p1[1] + i * dy)
        kN.append(k(D(i)))
        if i > 0:
            kS.append(k(D(i)))
            S.append(Solid)


class Parachute:

    def __init__(self):

        self.X = []
        self.Y = []
        self.KN = []
        self.KS = []
        self.S = []
        self.LNO = []
        self.LS0 = []
        self.p = []

    def create_parachute(self, Dd, Dv, Hg, Hb, Ls, Ns, Nb, Ng, Nd, Nv, Ed, Er, Es, Nsus, dp):
        theta = np.arcsin(Dd / 2 / Ls)
        pointsX = [-Ls * np.cos(theta), 0, Hb, Hb + Hg, Hb + Hg, Hb + Hg, Hb + Hg, Hb, 0, -Ls * np.cos(theta)]
        pointsY = [0, Dd / 2, Dd / 2, Dd / 2, Dv / 2, -Dv / 2, -Dd / 2, -Dd / 2, -Dd / 2, 0]

        k_sus = interpolate.interp1d([Dv - 0.01, Dd + 0.01], [Ns / Ls * (Es * Nsus * 4 / np.pi / Dd), Ns / Ls * (Es * Nsus * 4 / np.pi / Dd)], 'linear')
        k_c = interpolate.interp1d([Dv - 0.01, Dd + 0.01], [Nd * 2 / (Dd - Dv) * (1 / np.pi / Dd * (np.pi * Dv * Ed + Nsus * (Er + Es))), Nd * 2 / (Dd - Dv) * (1 / np.pi / Dd * (np.pi * Dd * Ed + Nsus * (Er + Es)))], 'linear')
        k_v = interpolate.interp1d([Dv - 0.01, Dd + 0.01], [Nv / Dv * ((Es + Er) * Nsus * 4 / np.pi / Dd), Nv / Dv * ((Es + Er) * Nsus * 4 / np.pi / Dd)], 'linear')
        k_g = interpolate.interp1d([Dv - 0.01, Dd + 0.01], [Ng / Hg * (Es * Nsus * 4 / np.pi / Dd), Ng / Hg * (Es * Nsus * 4 / np.pi / Dd)], 'linear')
        k_b = interpolate.interp1d([Dv - 0.01, Dd + 0.01], [Nb * 2 / Hb * (1 / np.pi / Dd * (np.pi * Dv * Ed + Nsus * (Er + Es))), Nb * 2 / Hb * (1 / np.pi / Dd * (np.pi * Dd * Ed + Nsus * (Er + Es)))], 'linear')
        k = [k_sus, k_b, k_g, k_c, k_v, k_c,
             k_g, k_b, k_sus]
        X = []
        Y = []
        LN0 = []
        LS0 = []
        KN = []
        KS = []
        S = []

        create_panel([pointsX[0], pointsY[0]], [pointsX[1], pointsY[1]], Ns, k[0], k[-1], k[1], X, Y, KN, KS, S, 0, [Dd, Dd], end=False)
        create_panel([pointsX[1], pointsY[1]], [pointsX[2], pointsY[2]], Nb, k[1], k[0], k[2], X, Y, KN, KS, S, 1, [Dd, Dd], end=False)
        create_panel([pointsX[2], pointsY[2]], [pointsX[3], pointsY[3]], Ng, k[2], k[1], k[3], X, Y, KN, KS, S, 0, [Dd, Dd], end=False)
        create_panel([pointsX[3], pointsY[3]], [pointsX[4], pointsY[4]], Nd, k[3], k[2], k[4], X, Y, KN, KS, S, 1, [Dd, Dv], end=False)
        create_panel([pointsX[4], pointsY[4]], [pointsX[5], pointsY[5]], Nv, k[4], k[3], k[5], X, Y, KN, KS, S, 0, [Dd, Dd], end=False)
        create_panel([pointsX[5], pointsY[5]], [pointsX[6], pointsY[6]], Nd, k[5], k[4], k[6], X, Y, KN, KS, S, 1, [Dd, Dv], end=False)
        create_panel([pointsX[6], pointsY[6]], [pointsX[7], pointsY[7]], Ng, k[6], k[5], k[7], X, Y, KN, KS, S, 0, [Dd, Dd], end=False)
        create_panel([pointsX[7], pointsY[7]], [pointsX[8], pointsY[8]], Nb, k[7], k[6], k[8], X, Y, KN, KS, S, 1, [Dd, Dd], end=False)
        create_panel([pointsX[8], pointsY[8]], [pointsX[9], pointsY[9]], Ns, k[8], k[7], k[0], X, Y, KN, KS, S, 0, [Dd, Dd], end=True)

        for ind in range(len(X)):
            i = ind % (len(X))
            ip = (ind + 1) % (len(X))
            im = (ind - 1) % (len(X))
            dxS = X[i] - X[im]
            dyS = Y[i] - Y[im]
            dxN = X[i] - X[ip]
            dyN = Y[i] - Y[ip]
            LN0.append((dxN ** 2 + dyN ** 2) ** 0.5)
            LS0.append((dxS ** 2 + dyS ** 2) ** 0.5)

        self.X = X
        self.Y = Y
        self.LN0 = LN0
        self.LS0 = LS0
        self.S = S
        self.KN = KN
        self.KS = KS
        self.dp = np.ones(len(X)) * dp

    def compute_elastic_force(self):
        X = np.array(self.X)
        Y = np.array(self.Y)
        LN0 = np.array(self.LN0)
        LS0 = np.array(self.LS0)
        KN = np.array(self.KN)
        KS = np.array(self.KS)
        LXN = X - np.roll(X, -1, 0)
        LXS = X - np.roll(X, 1, 0)
        LYN = Y - np.roll(Y, -1, 0)
        LYS = Y - np.roll(Y, 1, 0)

        LN = (LXN ** 2 + LYN ** 2) ** 0.5
        LS = (LXS ** 2 + LYS ** 2) ** 0.5
        dLN = LN - LN0
        dLS = LS - LS0

        FkNX = KN * dLN * LXN / LN
        FkNY = KN * dLN * LYN / LN
        FkSX = KS * dLS * LXS / LS
        FkSY = KS * dLS * LYS / LS

        FkN = np.array([FkNX, FkNY])
        FkS = np.array([FkSX, FkSY])
        return FkN, FkS

    def simulate(self, kr, dp, Nt, T):
        X = np.array(self.X)
        Y = np.array(self.Y)
        S = np.array(self.S, bool)
        Vx = np.zeros(len(X))
        Vy = np.zeros(len(Y))
        dt = T / Nt
        density = 10
        C = 5

        for i in range(Nt):

            X1 = 0.5 * (X + np.roll(X, 1, 0))
            X2 = 0.5 * (X + np.roll(X, -1, 0))
            Y1 = 0.5 * (Y + np.roll(Y, 1, 0))
            Y2 = 0.5 * (Y + np.roll(Y, -1, 0))
            FkN, FkS = self.compute_elastic_force()
            FkN[:, 0] = FkN[:, -1] = 0
            FkS[:, 0] = FkS[:, -1] = 0
            sumFX = np.zeros(len(X))
            sumFY = np.zeros(len(X))

            sumFX -= FkN[0]
            sumFX -= FkS[0]
            sumFY -= FkN[1]
            sumFY -= FkS[1]

            dX = X2 - X1
            dY = Y2 - Y1
            mod_n = (dX ** 2 + dY ** 2) ** 0.5
            n = np.array([-dY / mod_n, dX / mod_n])
            Surf = mod_n
            # sumFX[S] += n[0][S] * dp * Surf[S]
            # sumFY[S] += n[1][S] * dp * Surf[S]
            sumFX[S] += n[0][S] * np.array(self.dp)[S] * Surf[S]
            sumFY[S] += n[1][S] * np.array(self.dp)[S] * Surf[S]

            sumFX -= C * Vx
            sumFY -= C * Vy
            sumFX[0] = sumFX[-1] = 0
            sumFY[0] = sumFY[-1] = 0
            Mass = density * Surf
            Vx += sumFX / Mass * dt
            Vy += sumFY / Mass * dt
            X += Vx * dt
            Y += Vy * dt
            self.X = X
            self.Y = Y
            if i % 50000 == 0:
                self.plotStrain()
                Xdisp = list(X.copy())
                Xdisp.append(X[0])
                Ydisp = list(Y.copy())
                Ydisp.append(Y[0])
                notS = [not elem for elem in S]
                # x = np.linspace(self.Lx, self.Rx, self.dimensions[1])
                # y = np.linspace(self.Ly, self.Ry, self.dimensions[0])
                # plt.imshow(np.flip(self.p(x, y).T, 0), extent=[self.Lx, self.Rx, self.Ly, self.Ry])
                # plt.colorbar(orientation='horizontal')
                plt.plot(Xdisp, Ydisp, linewidth=1)
                plt.scatter(list(compress(X, S)), list(compress(Y, S)), marker="o", s=5)
                plt.scatter(list(compress(X, notS)), list(compress(Y, notS)), marker="o", s=5)
                plt.gca().set_aspect('equal')
                #plt.grid()
                plt.show()
                print("Computing iteration", i, "!")
        titles = ["X", "Y", "S"]
        matrix = np.array([X, Y, S]).T
        self.saveCSV("parachute", titles, matrix, True)

    def saveSTL(self, thickness):
        X = self.X
        Y = self.Y
        S = self.S

        vertices = []
        indices = []

        for i in range(len(X)):
            if S[i]:
                vertices.append([X[i], Y[i], 0])
                indices.append(i)
        for i in range(len(X)):
            if S[i]: vertices.append([X[i], Y[i], thickness])

        offset = int(0.5 * (len(vertices)))
        vertices = np.array(vertices)
        faces = []
        ind = 0
        for i in range(offset - 1):
            t11 = i
            t12 = i + 1
            t13 = i + offset
            t21 = i + 1
            t22 = i + offset
            t23 = i + 1 + offset
            if S[indices[i]] == True and S[indices[i] + 1] == True:
                faces.append([t11, t12, t13])
                faces.append([t21, t22, t23])

        faces = np.array(faces)

        parachute = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                parachute.vectors[i][j] = vertices[f[j], :]

        parachute.save('parachute.stl')

        for i in range(offset - 1):
            if S[indices[i]] == True and S[indices[i] + 1] == True:
                faces = []
                vertices_p = [vertices[i], vertices[i+1], vertices[i + offset], vertices[i + offset + 1]]
                vertices_p = np.array(vertices_p)
                t11 = 0
                t12 = 1
                t13 = 2
                t21 = 1
                t22 = 2
                t23 = 3
                faces.append([t11, t12, t13])
                faces.append([t21, t22, t23])
                faces = np.array(faces)

                panel = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for k, f in enumerate(faces):
                    for j in range(3):
                        panel.vectors[k][j] = vertices_p[f[j], :]
                panel.save('STL_Files/panel' + str(ind) + '.stl')
                ind += 1

    def importParachute(self, file):
        data = np.genfromtxt(file, delimiter=",", skip_header=1).T
        self.X = list(data[0])
        self.Y = list(data[1])
        self.S = list(data[2])

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

    def importPressure(self, file, dimensions, limits):
        data = np.genfromtxt(file, skip_header=23, skip_footer=23)
        p = data.reshape((dimensions[1], dimensions[0]))
        Lx = limits[0][0]
        Rx = limits[0][1]
        Ly = limits[1][0]
        Ry = limits[1][1]
        x = np.linspace(Lx, Rx, dimensions[0])
        y = np.linspace(Ly, Ry, dimensions[1])
        distance = 0.05
        self.p = interpolate.RectBivariateSpline(x, y, p.T)
        x = np.linspace(Lx, Rx, dimensions[0])
        y = np.linspace(Ly, Ry, dimensions[1])
        dxp = (Rx - Lx) / dimensions[1]
        dyp = (Ry - Ly) / dimensions[0]
        self.dimensions = dimensions
        self.Lx = Lx
        self.Rx = Rx
        self.Ly = Ly
        self.Ry = Ry
        self.dp = np.ones(len(self.X))
        X = np.array(self.X)
        Y = np.array(self.Y)
        x_up = [0]
        y_up = [0]
        x_d = [0]
        y_d = [0]

        X1 = 0.5 * (X + np.roll(X, 1, 0))
        X2 = 0.5 * (X + np.roll(X, -1, 0))
        Y1 = 0.5 * (Y + np.roll(Y, 1, 0))
        Y2 = 0.5 * (Y + np.roll(Y, -1, 0))
        FkN, FkS = self.compute_elastic_force()
        FkN[:, 0] = FkN[:, -1] = 0
        FkS[:, 0] = FkS[:, -1] = 0
        dX = X2 - X1
        dY = Y2 - Y1
        mod_n = (dX ** 2 + dY ** 2) ** 0.5
        n = np.array([-dY / mod_n, dX / mod_n])
        p1 = self.p.ev(X + distance * n[0], Y + distance * n[1])
        p2 = self.p.ev(X - distance * n[0], Y - distance * n[1])
        self.dp = p2 - p1

        plt.clf()
        plt.plot(np.arange(0, len(self.X), 1), self.dp)
        plt.plot(np.arange(0, len(self.X), 1), np.array(self.S) * 100)
        plt.show()

    def plotStrain(self):

        X = np.array(self.X)
        Y = np.array(self.Y)
        print(self.X)

        LXN = X - np.roll(X, -1, 0)
        LYN = Y - np.roll(Y, -1, 0)
        LN0 = np.array(self.LN0)

        strain = (LXN**2 + LYN**2)**0.5 / LN0 - 1

        plt.plot(np.arange(0, len(X), 1), strain)
        plt.show()

    def Cd(self, rho, v_inf, Dd):
        X = np.array(self.X)
        Y = np.array(self.Y)
        X1 = 0.5 * (X + np.roll(X, 1, 0))
        X2 = 0.5 * (X + np.roll(X, -1, 0))
        Y1 = 0.5 * (Y + np.roll(Y, 1, 0))
        Y2 = 0.5 * (Y + np.roll(Y, -1, 0))
        dX = X2 - X1
        dY = Y2 - Y1
        mod_n = (dX ** 2 + dY ** 2) ** 0.5
        n = np.array([-dY / mod_n, dX / mod_n])

        dS = (dX**2 + dY**2)**0.5
        S = np.array(self.S, bool)
        Cd = np.sum((self.dp * dY / (0.5 * rho * v_inf**2))[S]) / Dd
        return Cd

    def Normals(self):
        X = np.array(self.X)
        Y = np.array(self.Y)
        S = np.array(self.S, bool)

        X1 = 0.5 * (X + np.roll(X, 1, 0))
        X2 = 0.5 * (X + np.roll(X, -1, 0))
        Y1 = 0.5 * (Y + np.roll(Y, 1, 0))
        Y2 = 0.5 * (Y + np.roll(Y, -1, 0))
        dX = X2 - X1
        dY = Y2 - Y1
        mod_n = (dX ** 2 + dY ** 2) ** 0.5
        n = np.array([-dY / mod_n, dX / mod_n])
        Sm = np.roll(S, 1, 0)
        e1x = n[0][(S & Sm)]
        e1y = np.zeros(e1x.shape)
        e1z = n[1][(S & Sm)]
        e2x = np.zeros(e1x.shape)
        e2y = np.ones(e1x.shape)
        e2z = np.zeros(e1x.shape)
        return np.array([e1x, e1y, e1z, e2x, e2y, e2z])



#
# D_disk = 1.0036
# D_vent = 0.1
# H_gap = 0.062
# # H_gap = 0.055323
# H_band = 0.293
# # H_band = 0.46
# L_sus = 1.328
#
# k_disk = 39275  # Nylon F111 canopy
# k_band = 39275  # Nylon F111 canopy
# k_sus = 84034.5  # Twaron CopSub
# k_reinf = 84034.5  # Twaron CopSub
#
# N_sus = 6
#
# Num_sus = 8
# Num_band = 10
# Num_gap = 5
# Num_disk = 20
# Num_vent = 5
#
# Nt = 200000
# T = 2
#
# d_pressure = 45
#
# par = Parachute()
# par.create_parachute(D_disk, D_vent, H_gap, H_band, L_sus, Num_sus, Num_band, Num_gap, Num_disk, Num_vent, k_disk, k_reinf, k_sus, N_sus, d_pressure)
# par.importParachute("parachute.csv")
# norm = par.Normals()
# OpenFOAM.createPorosityProperties(norm, [0, 0, 0], [150, 10000, 150])
# OpenFOAM.createFvOptions(len(norm[0])+1)
# OpenFOAM.createTopoSet(len(norm[0])+1)
# par.importPressure("p", [800, 800], [[-3, 3], [-3, 3]])
# print("The Cd is", par.Cd(1.225, 30, D_disk))
# par.simulate(k_reinf, d_pressure, Nt, T)
#
# par.saveSTL(thickness=1.0)
