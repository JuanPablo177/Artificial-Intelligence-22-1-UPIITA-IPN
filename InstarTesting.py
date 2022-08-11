from skimage import color, io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from win32com.client import Dispatch
import cv2

speak = Dispatch("SAPI.SpVoice").Speak

def Softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x) # only difference

def HardLims(n):
    a = np.where(n > 0, 1, -1)
    return a

def HardLim(n):
    a = np.where(n > 0, 1, 0)
    return a

def CortarImagen(sumaXFoC, umbralC):
    cadaCosa = [0]
    for i in range(len(sumaXFoC[0])):
        if (i < len(sumaXFoC[0])-1):
            if ((sumaXFoC[0][i+1]-sumaXFoC[0][i])>umbralC):
                cadaCosa.append(i)
                cadaCosa.append(i+1)
        else:
            cadaCosa.append(i)
    return cadaCosa

abecedario = ["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]

W = np.array(pd.read_csv('WFinal.csv'))
W = W[:,1:]

fil, B = W.shape
maxLongitud = B
b = -B*0.7

ima = cv2.cvtColor(cv2.imread('Poema.bmp'),cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(ima,cmap='gray')

ima2 = ((ima)<255).astype(int) # Binarizar a menores de 255

# ---------------------------------- Corte Renglones --------------------------
sumaXFilas = np.sum(ima2,axis=1)
plt.figure()
plt.plot(sumaXFilas)

indicesXFilas = np.where(sumaXFilas!=0)
cadaRenglon = CortarImagen(indicesXFilas, 1)


dim = (45,36)

for reng in range(0, len(cadaRenglon), 2):
# for reng in range(2, 12, 2):
    corte_1 = indicesXFilas[0][cadaRenglon[reng]]
    corte_2 = indicesXFilas[0][cadaRenglon[reng+1]]
        
    renglonActual = ima2[corte_1:corte_2,:]
    plt.figure()
    plt.imshow(renglonActual)
    plt.close('all')
    # ---------------------------------- Corte Columna de renglones -----------
    sumaXColumnas = np.sum(renglonActual,axis=0)
    plt.figure()
    plt.plot(sumaXColumnas)
    
    indicesXCol = np.where(sumaXColumnas!=0)
    corte_11 = indicesXCol[0][0]
    corte_22 = indicesXCol[0][len(indicesXCol[0])-1]
    
    renglonFinal = renglonActual[:,corte_11:corte_22]
    plt.figure()
    plt.imshow(renglonFinal)
    
    # ---------------------------------- Corte Columna de letras --------------
    sumaXColumnasLetra = np.sum(renglonFinal,axis=0)
    plt.figure()
    plt.plot(sumaXColumnasLetra)
        
    indicesXColLetra = np.where(sumaXColumnasLetra!=0)
    cadaLetra = CortarImagen(indicesXColLetra, 1)
    
    P = []
    
    for i in range(0, len(cadaLetra), 2):
        corte_111 = indicesXColLetra[0][cadaLetra[i]]
        corte_222 = indicesXColLetra[0][cadaLetra[i+1]]
        
        letraActual = renglonFinal[:,corte_111:corte_222]
        
        # ---------------------------------- Corte Renglon ------------------------
        sumaXFilasLetra = np.sum(letraActual,axis=1)
        
        indicesXFilasLetra = np.where(sumaXFilasLetra!=0)
        cadaRenglonLetra = CortarImagen(indicesXFilasLetra, 1)
        
        corte_1111 = indicesXFilasLetra[0][cadaRenglonLetra[0]]
        corte_2222 = indicesXFilasLetra[0][cadaRenglonLetra[len(cadaRenglonLetra) - 1]]
                
        matrizLetra = letraActual[corte_1111:corte_2222,:]
    
        # -------------------------------------------------------------------------
        matrizLetra_2 = cv2.resize(np.uint8(matrizLetra), dim, interpolation=cv2.INTER_AREA)

        vectorLetra = np.ravel(matrizLetra_2, order='F')
        P.append(np.where(vectorLetra < 0.5, -1, vectorLetra))
        
    #----------------------------------- Instar ---------------------------
    word = ''
    for n in range(len(P)):
        esLetra = []
        
        for j in range(len(W)):
            w = np.reshape(W[j,:],(1,B))
            p = np.reshape(P[n],(B,1))
            a = float(w.dot(p))   
            esLetra.append(a)

        esLetra = np.array(esLetra)
        esLetraF = np.argmax(esLetra)
        
        letraFinal = abecedario[0][esLetraF]
        if (len(word) != 0):
            if(letraFinal == 'I' and (word[len(word)-1] == 'a' or word[len(word)-1] == 'e' or word[len(word)-1] == 'i' or word[len(word)-1] == 'o' or word[len(word)-1] == 'u' or word[len(word)-1] == 'l' or word[len(word)-1] == 'y')):
                letraFinal = 'l'
        
        word = word + letraFinal
        
    print(word)
    speak(word)
    

    



