import numpy as np
import math
import matplotlib.pyplot as plt

def computeRow(x):
    return (x[1][0]*x[0][1] - x[0][0]*x[1][1])/x[1][0]

def routh_array(f):
    global flag
    n = len(f)-1
    s1 = np.zeros((math.ceil(len(f)/2) + 1)) #1st two rows of the Routh array
    s2 = np.zeros((len(s1)))
    r_array = np.zeros((n+1, len(s1)))

    i = 0
    j = 0
    while i <= n:
        s1[j] = f[i]
        if i < n:
            s2[j] = f[i+1]
        i += 2
        j += 1

    r_array[0] = s1
    r_array[1] = s2
    index = 0

    for i in range(2, n+1):
        if r_array[i-1][0] == 0:
            if np.all(r_array[i-1] == 0):
                index = i-2
                for k in range(len(r_array[0])):
                    r_array[index+1][k] = ((n-index-2*k)*r_array[index][k])
            else:
                r_array[i-1][0] = 1e-4 #some small number

        for j in range(0, len(s1)-1):
            r_array[i][j] = computeRow([[r_array[i-2][0], r_array[i-2][j+1]], [r_array[i-1][0], r_array[i-1][j+1]]])        
        
        if r_array[i-1][0]*r_array[i-2][0] < 0:
            flag = 0

    if flag != 0 and index != 0:
        flag = 2     

    return r_array

#f holds the coefficients of the characteristic function
f = np.array([1, 2, 8, 12, 20, 16, 16]) #corresponding to H(s) = s^6 + 2s^5 + 8s^4 + 12s^3 + 20s^2 + 16s + 16
flag = 1
arr = routh_array(f)
if flag == 1:
    print("Stable system")
elif flag == 2:
    print("Marginally Stable system")
else:
    print("Unstable system")

print("Routh array:\n", arr)


         

