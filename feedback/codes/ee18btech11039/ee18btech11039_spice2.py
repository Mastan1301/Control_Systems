import numpy as np  
from matplotlib import pyplot as plt  

#if using termux
import subprocess
import shlex
#end if

T = np.loadtxt('./codes/ee18btech11039/ratio.dat')
print(T.shape)
plt.figure()
plt.plot(T[:, 0], T[:, 1])
plt.xlabel("f");
plt.ylabel("$V_{o}/V_{s}$")
plt.grid()

#if using termux
plt.savefig('./figs/ee18btech11039/spice_2.pdf')
plt.savefig('./figs/ee18btech11039/spice_2.eps')
subprocess.run(shlex.split("termux-open ./figs/ee18btech11039/spice_2.pdf"))
#else
#plt.show()
