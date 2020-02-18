import control
import matplotlib.pyplot as plt 
import numpy as np 

#Plotting the inverse laplace transform of a transfer function
k = [-3, 1, 3]
for j in k:
    num = [1]
    den = [1, j, j+2, 3]
    H = control.TransferFunction(num, den)
    p = H.pole()
    x = [p.real for i in p]
    y = [p.imag for i in p]
    plt.scatter(x, y, color = 'red')
    plt.xlabel('$\sigma$')
    plt.ylabel('$\omega$')
    plt.title('For k = '+str(j))
    plt.grid()
    plt.savefig('k='+str(j)+'.jpg')    
    plt.show()