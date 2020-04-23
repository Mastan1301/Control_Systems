import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#if using termux
import subprocess
import shlex
#end if

def CalculatePolesAndZeros(num, den): #returns an array of zeros and poles
	pz = []
	pz.append(np.flip(np.roots(num)))
	pz.append(np.flip(np.roots(den)))
	return pz

def asymptoticPlotMag(x, pz): # returns the values for asymptotic Bode plot
    res = np.zeros((len(x)))
    res[0] = 20*np.log10(1e6) # poles at origin
    val = res[0]
    slope = 0
    j, k = 0, 0
    for i in range(1, len(x)):
        while j < len(pz[0]) and x[i-1] == abs(pz[0][j]): # check for multiple poles/ zeros
            slope += 20
            j += 1
        while k < len(pz[1]) and x[i-1] == abs(pz[1][k]):
            slope -= 20
            k += 1

        if x[i-1] == 0.0: # avoiding division by zero
            x[i-1] = 1e-2
        if x[i] == 0.0:
            x[i] = 1e-2

        if(x[i] != x[i-1]): 
            val += slope*np.log10(x[i]/x[i-1])

        res[i] = val

    return res

def asymptoticPlotPhase(x, pz):
    res = np.zeros((len(x)))
    res[0] = 90
    val = res[0]
    slope = 0
    temp = 1

    while(x[temp] == 0.0):
        x[temp] = 1e-2
        res[temp] = val
        temp += 1

    j, k = 0, temp

    for i in range(temp, len(x)-1):
        while j < len(pz[0]) and x[i-1] == abs(pz[0][j]/10): # check for multiple poles/ zeros
            slope += 45
            j += 1

        while k < len(pz[1]) and x[i-1] == abs(pz[1][k]/10):
            slope -= 45
            k += 1

        if(x[i] != x[i-1]): 
            val += slope*np.log10(x[i]/x[i-1])

        res[i] = val

    res[len(x)-1] = 90
    return res 

num = [0.2*0.025, 0.225, 1]
den = [0.005*0.001, 0.006, 1, 0, 0, 0]
s1 = signal.lti(num,den)
pz = CalculatePolesAndZeros(num, den)
print("Zeros: ", pz[0])
print("Poles: ", pz[1])
w, mag, phase = signal.bode(s1)

x = []
for i in range(len(pz)):
    for j in range(len(pz[i])):
        x.append(abs(pz[i][j]))

x.sort() # x-axis for drawing the theoretical asymptotic magnitude plot
x1 = [i/10 for i in x]

x.append(w[-1]) # appending the upper limit of w
x1.append(w[-1]) # appending the upper limit of w

y = asymptoticPlotMag(x, pz)
phi = asymptoticPlotPhase(x1, pz)

plt.figure()
plt.subplot(2, 1, 1)
plt.xlabel("$w$")
plt.ylabel("20$log_{10}(|H(jw)|$")
plt.title("Magnitude Plot")
plt.semilogx(w, mag)    # Using in-built function

plt.semilogx(x, y) # Theoretical plot
plt.legend(["Using in-built function" , "Asymptotic Plot"])
plt.grid()

plt.subplot(2, 1, 2)
plt.xlabel("$w$")
plt.ylabel("$\phi(j\omega)$")
plt.title("Phase Plot")
plt.semilogx(w, phase)    # Using in-built function

plt.semilogx(x1, phi) # Theoretical plot
plt.legend(["Using in-built function" , "Asymptotic Plot"])
plt.grid()

plt.show()

''' #if using termux
plt.savefig('./figs/ee18btech11001/ee18btech11001_2.pdf')
plt.savefig('./figs/ee18btech11001/ee18btech11001_2.eps')
subprocess.run(shlex.split("termux-open ./figs/ee18btech11001/ee18btech11001_2.pdf"))'''
#else
#plt.show() 