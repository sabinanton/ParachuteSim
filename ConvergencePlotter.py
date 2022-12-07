import numpy as np

import Parachute3D as par
import matplotlib.pylab as plt

########### Non-linear Materials ###################
plt.rcParams.update({'font.size': 15})
k = 0.2
RipNylon = par.Canopy_Material(2.7e6, 0.048, [16615 * (1 - k), 67021 * (1 - k)], [16615 * k / 2, 67021 * k / 2], 0.09e-3, 1)
Aramid = par.Canopy_Material(3.4e6, 0.03, [211985, 489633], [500], 0.43e-3, 1)
Spectra = par.Suspension_Material(1.1e7, 0.0027, [3064.2], 4e-3)

Disks = [[0.1 / 2, 1.0036 / 2]]
Bands = [[0.062, 0.062 + 0.293]]

Suspension_Length = 1.328
Num_Suspension = 18
Num_Gores = 6
Reinforcement_Width = 25e-3
# Disk_Resolution = [120]
Disk_Resolution = [50]
Band_Resolution = [40]
Angular_Resolution = 256
Sus_Resolution = [10]

initial_pressure = 470


num_iter = 7
parachutes = []
for i in range(num_iter):

    parachutei = par.Parachute(Disks, Bands, Num_Suspension, Num_Gores, Suspension_Length, Reinforcement_Width, Disk_Resolution, Band_Resolution, Sus_Resolution, Angular_Resolution,
                      RipNylon,
                      Spectra, Aramid, initial_pressure, 1.0036 / 2)
    parachutei.importParachute('StratosIV_Iter'+str(i))
    parachutes.append(parachutei)

X0 = parachutes[-1].Disks[0].X
Y0 = parachutes[-1].Disks[0].Y
Z0 = parachutes[-1].Disks[0].Z

Error = []
Iter = []

for i in range(num_iter):
    Iter.append(i)
    Xi = parachutes[i].Disks[0].X
    Yi = parachutes[i].Disks[0].Y
    Zi = parachutes[i].Disks[0].Z

    e = ((Xi - X0)**2 + (Yi - Y0)**2 + (Zi - Z0)**2)**0.5
    error = np.mean(e)
    Error.append(error)


fig2 = plt.figure(figsize=(8, 5))
ax = fig2.add_subplot()
plt.suptitle(r"Drag Force vs. Time")

ax.plot(Iter, Error, linewidth=1, marker='o')
ax.set_xlabel(r"Iterations [-]")
ax.set_ylabel(r"Mean Shape Error [m]")
# ax.set_ylim([0, 2.5])
ax.grid()
plt.savefig("ConvergenceAnalysis.png", dpi=1000)
plt.close(fig2)