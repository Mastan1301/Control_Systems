import numpy as np
from scipy import signal
import control
import matplotlib.pyplot as plt

#if using termux
import subprocess
import shlex
#end if

num = [0.2*0.025, 0.225, 1]
den = [0.005*0.001, 0.006, 1, 0, 0, 0]
G = signal.lti(num,den)
w, mag, phase = signal.bode(G)
sys = control.tf(num, den)
gm, pm, wg, wp = control.margin(sys)
gm = -20*np.log10(gm)
    
print('Gain Margin: ', gm, ', Gain cross-over frequency: ', wg)
print('Phase Margin: ', pm, ', Phase cross-over frequency: ', wp)

#Magnitude plot
plt.figure()
plt.subplot(2, 1, 1) 
plt.xlabel("$\omega$")
plt.ylabel("20$log_{10}(|G(j\omega)|$")
plt.title("Magnitude Plot")
plt.semilogx(w, mag) 
plt.plot(wg, gm, 'o')
plt.plot(wp, 0, 'o')
plt.text(wg+0.5, gm+2, '({}, {})'.format(round(wg, 2), round(gm, 2)))
plt.text(wp+0.5, 0+2, '({}, {})'.format(round(wp, 2), 0))
plt.grid()

# Phase plot
plt.subplot(2, 1, 2) 
plt.xlabel("$\omega$")
plt.ylabel("$\phi(j\omega)$")
plt.title("Phase Plot")
plt.semilogx(w, phase)  
plt.plot(wp, pm+180, 'o')
plt.plot(wg, 180, 'o')
plt.text(wp+0.5, pm+178, '({}, {})'.format(round(wp, 2), round(pm+180, 2)))
plt.text(wg+0.5, 178, '({}, {})'.format(round(wg, 2), 180))
plt.grid()

#if using termux
plt.savefig('./figs/ee18btech11039/ee18btech11039.pdf')
plt.savefig('./figs/ee18btech11039/ee18btech11039.eps')
subprocess.run(shlex.split("termux-open ./figs/ee18btech11039/ee18btech11039.pdf"))
#else
#plt.show() 
