import numpy as np  
from matplotlib import pyplot as plt  

#if using termux
import subprocess
import shlex
#end if

din, dout = np.loadtxt('./codes/ee18btech11039/input.dat'), np.loadtxt('./codes/ee18btech11039/output.dat')  

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(din[:, 0], din[:, 1])
plt.xlabel("t")
plt.ylabel("$V_{s}(t)$")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(dout[:, 0], dout[:, 1])
plt.xlabel("t")
plt.ylabel("$V_{o}(t)$")
plt.grid()

#if using termux
plt.savefig('./figs/ee18btech11039/spice_1.pdf')
plt.savefig('./figs/ee18btech11039/spice_1.eps')
subprocess.run(shlex.split("termux-open ./figs/ee18btech11039/spice_1.pdf"))
#else
#plt.show()
