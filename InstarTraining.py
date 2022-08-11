import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from win32com.client import Dispatch
from math import sqrt
import cv2

def HardLim(n):
    a = np.where(n > 0, 1, 0)
    return a

def HardLims(n):
    a = np.where(n > 0, 1, -1)
    return a

def DistEuclidiana(p1,p2):
    suma = 0
    for i in range(len(p1)):
        suma += (p1[i] - p2[0][i])**2
    d =sqrt(suma)
    return d


P = np.array(pd.read_csv('P.csv'))
P = P[:,1:]

# P = np.where(P == 0, -1, P)


fil, B = P.shape
P0 =  1
W0 = B + B*0.1
b = -B
alpha = 0.1
epocas = 20000
W = np.random.rand(1,B)

# # ----------------------------- Con for -----------------
# PesosS = []
# for i in range(fil):
#     for j in range(epocas):
#         a = hardLim((W0*P0) + W.dot(P[i,:].T) + b)
#         W = (1-alpha)*W + alpha*(P[i,:].T)
        
#     PesosS.append(W)


# # ----------------------------- Con while
PesosS = []
for i in range(fil):
    error = 1000
    while(error > 0.000001):
        a = HardLim((W0*P0) + W.dot(P[i,:].T) + b)
        W = (1-alpha)*W + alpha*(P[i,:].T)

        error = DistEuclidiana(P[i,:],W)
        print(error)
    PesosS.append(W)



WFinal = np.reshape(PesosS, (fil, B))
WFinal = np.where(WFinal > 0.5, 1, 0)


if((WFinal == P).all()):
    print("La matriz de Pesos (W) SI es identica a los Patrones (P)")
    # # ----------------------------- Guardar pesos de salida -----------------
    df = pd.DataFrame(WFinal)
    df.to_csv('WFinal.csv')  
else:
    print("La matriz de Pesos (W) NO es identica a los Patrones (P)")
  
    
  