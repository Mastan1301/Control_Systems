import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#if using termux
import subprocess
import shlex
#end if

def CalculatePolesAndZeros(num, den):
	pz = []
	pz.append(np.flip(np.roots(num)))
	pz.append(np.flip(np.roots(den)))
	return pz

def asymptoticPlot(x, pz):
    arr = np.zeros((len(x)))
    arr[0] = 20*np.log10(1e6) # poles at origin
    val = arr[0]
    slope = 0
    j, k = 0, 0
    for i in range(1, len(x)):
        while j < len(pz[0]) and x[i] == abs(pz[0][j]): # check for multiple poles/ zeros
            slope += 20
            j += 1
        while k < len(pz[1]) and x[i] == abs(pz[1][k]):
            slope -= 20
            k += 1

        temp1 = x[i-1]
        temp2 = x[i]
        if temp1 == 0.0:
            temp1 = 1e-2
        if temp2 == 0.0:
            temp2 = 1e-2

        print(temp1, temp2)
        val += slope*np.log10(temp2/temp1)
        arr[i] = val

    return arr


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

x.sort()
y = asymptoticPlot(x, pz)
print(y)

plt.figure()
plt.xlabel("f")
plt.ylabel("H(f)")
plt.title("Bode Plot")
plt.semilogx(w, mag)    # Bode magnitude plot'''

plt.plot(x, y)
plt.legend(["Using in-built function" , "Asymptotic Plot"])
plt.grid() 
plt.show()
''' if using termux
plt.savefig('./figs/ee18btech11001/ee18btech11001_2.pdf')
plt.savefig('./figs/ee18btech11001/ee18btech11001_2.eps')
subprocess.run(shlex.split("termux-open ./figs/ee18btech11001/ee18btech11001_2.pdf"))
#else
#plt.show() '''