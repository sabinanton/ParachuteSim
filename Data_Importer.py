import numpy as np
from matplotlib import pyplot as plt

# data = np.genfromtxt('loadcellstratosiv.txt', skip_header=1).T
# data[0] = data[0]
# data[1] = data[1] / 10**6
# data[0] = -np.where(np.abs(data[0]) > 10000, 0, data[0])
# data[1] = np.where(np.abs(data[1]) < 0, 0, data[1])
# plt.plot(data[1][((data[1] > 10.5) & (data[1] < 210))], data[0][((data[1] > 10.5) & (data[1] < 210))])
# #plt.plot(data[1], data[0])
# plt.grid()
# plt.show()


plt.rcParams.update({'font.size': 14})
Walrus = [598.9389, 415.9298, 295.6587, 166.308, 73.9147]
StratosIV = [396.05242, 275.0364, 173.2345, 97.444, 43.309]
RingSail = [213.804, 100.269, 26.247]
Cd = Walrus

fig2 = plt.figure(figsize=(8, 6))
ax = fig2.add_subplot()
plt.suptitle(r"Drag Force vs. Time - WALRUS")
ax.set_xlabel(r"Time [s]")
ax.set_ylabel(r"Drag Force [N]")

data = np.genfromtxt('TestRawData/Test13.csv').T
time = np.linspace(0, 400, len(data))
ax.plot(time, data, linewidth=0.55, label='Experimental', color='gray')

time_30 = time[((time > 16.8) & (time < 84.5))]
Cd_Walrus_30 = np.ones(time_30.shape) * Cd[0]
ax.plot(time_30, Cd_Walrus_30, linewidth=2.5, label='Simulated 30 m/s', linestyle='--')

time_25 = time[((time > 85) & (time < 145))]
Cd_Walrus_25 = np.ones(time_25.shape) * Cd[1]
ax.plot(time_25, Cd_Walrus_25, linewidth=2.5, label='Simulated 25 m/s', linestyle='--')

time_20 = time[((time > 145) & (time < 219))]
Cd_Walrus_20 = np.ones(time_20.shape) * Cd[2]
ax.plot(time_20, Cd_Walrus_20, linewidth=2.5, label='Simulated 20 m/s', linestyle='--')

time_15 = time[((time > 223) & (time < 283))]
Cd_Walrus_15 = np.ones(time_15.shape) * Cd[3]
ax.plot(time_15, Cd_Walrus_15, linewidth=2.5, label='Simulated 15 m/s', linestyle='--')

time_10 = time[((time > 283) & (time < 353))]
Cd_Walrus_10 = np.ones(time_10.shape) * Cd[4]
ax.plot(time_10, Cd_Walrus_10, linewidth=2.5, label='Simulated 10 m/s', linestyle='--')

ax.legend()
plt.savefig("WALRUS_Drag_SteadyState", dpi=1000)
plt.close(fig2)

fig2 = plt.figure(figsize=(8, 6))
ax = fig2.add_subplot()
plt.suptitle(r"Drag Force vs. Time - Ringsail")
ax.set_xlabel(r"Time [s]")
ax.set_ylabel(r"Drag Force [N]")
Cd = RingSail

data = np.genfromtxt('TestRawData/Test08.csv').T
time = np.linspace(0, 245, len(data))
ax.plot(time, data, linewidth=0.15, label='Experimental', color='gray')

time_30 = time[((time > 11) & (time < 147))]
Cd_Walrus_30 = np.ones(time_30.shape) * Cd[0]
ax.plot(time_30, Cd_Walrus_30, linewidth=2.5, label='Simulated 30 m/s', linestyle='--')

time_20 = time[((time > 152) & (time < 195))]
Cd_Walrus_20 = np.ones(time_20.shape) * Cd[1]
ax.plot(time_20, Cd_Walrus_20, linewidth=2.5, label='Simulated 20 m/s', linestyle='--')

time_10 = time[((time > 200) & (time < 239))]
Cd_Walrus_10 = np.ones(time_10.shape) * Cd[2]
ax.plot(time_10, Cd_Walrus_10, linewidth=2.5, label='Simulated 10 m/s', linestyle='--')

ax.legend()
# plt.show()
plt.savefig("Scottish_Drag_SteadyState", dpi=1000)
plt.close(fig2)

fig2 = plt.figure(figsize=(8, 6))
ax = fig2.add_subplot()
plt.suptitle(r"Drag Force vs. Time - Stratos IV")
ax.set_xlabel(r"Time [s]")
ax.set_ylabel(r"Drag Force [N]")
Cd = StratosIV

data = -np.genfromtxt('TestRawData/Test22.csv').T
data = np.where(data > 1300, 1300, data)
data = np.where(data < 0, 0, data) * 0.546215
time = np.linspace(0, 225, len(data))
ax.plot(time, data, linewidth=0.5, label='Experimental', color='gray')
# plt.show()

time_30 = time[((time > 31) & (time < 91))]
Cd_Walrus_30 = np.ones(time_30.shape) * Cd[0]
ax.plot(time_30, Cd_Walrus_30, linewidth=2.5, label='Simulated 30 m/s', linestyle='--')

time_25 = time[((time > 93) & (time < 133))]
Cd_Walrus_25 = np.ones(time_25.shape) * Cd[1]
ax.plot(time_25, Cd_Walrus_25, linewidth=2.5, label='Simulated 25 m/s', linestyle='--')

time_20 = time[((time > 135) & (time < 177))]
Cd_Walrus_20 = np.ones(time_20.shape) * Cd[2]
ax.plot(time_20, Cd_Walrus_20, linewidth=2.5, label='Simulated 20 m/s', linestyle='--')

time_15 = time[((time > 178) & (time < 214))]
Cd_Walrus_15 = np.ones(time_15.shape) * Cd[3]
ax.plot(time_15, Cd_Walrus_15, linewidth=2.5, label='Simulated 15 m/s', linestyle='--')

time_10 = time[((time > 216) & (time < 246))]
Cd_Walrus_10 = np.ones(time_10.shape) * Cd[4]
ax.plot(time_10, Cd_Walrus_10, linewidth=2.5, label='Simulated 10 m/s', linestyle='--')

ax.legend()
plt.savefig("StratosIV_Drag_SteadyState", dpi=1000)
plt.close(fig2)

data = np.genfromtxt('WALRUSTransientFinal.csv', delimiter=',', skip_header=1).T
time = data[0]
WALRUS_transient = data[1]
WALRUS_transient_e = np.genfromtxt('TestRawData/Test13.csv').T
time_e = np.linspace(0, 400, len(WALRUS_transient_e))
Map = ((time_e > 17.4) & (time_e < 19.2))
fig2 = plt.figure(figsize=(8, 6))
ax = fig2.add_subplot()
#plt.suptitle(r"Transient Drag Force vs. Time - WALRUS")
ax.set_xlabel(r"Time [s]")
ax.set_ylabel(r"Drag Force [N]")
ax.plot(time * 15, WALRUS_transient, linewidth=2, label='Simulation')
ax.plot(time_e[Map] - 17.4, WALRUS_transient_e[Map], linewidth=2, linestyle = '--', label='Experiment')
ax.legend()
plt.savefig("WALRUS_Transient", dpi=1000)

data = np.genfromtxt('StratosTransientFinal.csv', delimiter=',', skip_header=1).T
time = data[0]
Stratos_transient = data[1] * 1.1
Stratos_transient_e = -np.genfromtxt('TestRawData/Test22.csv').T
Stratos_transient_e = np.where(Stratos_transient_e > 1300, 1300, Stratos_transient_e)
Stratos_transient_e = np.where(Stratos_transient_e < 0, 0, Stratos_transient_e) * 0.546215
time_e = np.linspace(0, 225, len(Stratos_transient_e))
Map = ((time_e > 28.98) & (time_e < 29.5))
fig2 = plt.figure(figsize=(8, 6))
ax = fig2.add_subplot()
#plt.suptitle(r"Transient Drag Force vs. Time - Stratos IV")
ax.set_xlabel(r"Time [s]")
ax.set_ylabel(r"Drag Force [N]")
ax.plot(time * 4, Stratos_transient, linewidth=2, label='Simulation')
ax.plot(time_e[Map] - 28.98, Stratos_transient_e[Map], linewidth=2, linestyle = '--', label='Experiment')
ax.legend()
plt.savefig("Stratos_Transient", dpi=1000)

data = np.genfromtxt('RingSailTransient.csv', delimiter=',', skip_header=1).T
time = data[0]
Ringsail_transient = data[1]
Ringsail_transient_e = np.genfromtxt('TestRawData/Test08.csv').T
# Ringsail_transient_e = np.where(Ringsail_transient_e > 1300, 1300, Ringsail_transient_e)
# Ringsail_transient_e = np.where(Ringsail_transient_e < 0, 0, Ringsail_transient_e) * 0.546215
time_e = np.linspace(0, 245, len(Ringsail_transient_e))
Map = ((time_e > 10.66) & (time_e < 13.2))
fig2 = plt.figure(figsize=(8, 6))
ax = fig2.add_subplot()
#plt.suptitle(r"Transient Drag Force vs. Time - ADEPT Ringsail")
ax.set_xlabel(r"Time [s]")
ax.set_ylabel(r"Drag Force [N]")
ax.plot(time * 19, Ringsail_transient, linewidth=2, label='Simulation')
ax.plot(time_e[Map] - 10.66, Ringsail_transient_e[Map], linewidth=2, linestyle = '--', label='Experiment')
ax.legend()
plt.savefig("Ringsail_Transient", dpi=1000)