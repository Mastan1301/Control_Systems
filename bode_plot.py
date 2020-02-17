from scipy import signal
import matplotlib.pyplot as plt

num = [2] #numerator and denominator of the transfer function.
den = [1, 3, 2, 1, 2]
h = signal.lti(num, den)
w, mag, phase = signal.bode(h)

plt.figure()
plt.semilogx(w, mag)
plt.title("Magnitude plot")
plt.xlabel("w")
plt.ylabel("$|H(s)|$")

plt.figure()
plt.semilogx(w, phase)
plt.title("Phase plot")
plt.xlabel("w")
plt.ylabel("$\phi(w)$")
plt.show()
