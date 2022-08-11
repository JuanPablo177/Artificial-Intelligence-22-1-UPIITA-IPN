import numpy as np
import matplotlib.pyplot as plt
import cv2

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
plt.close('all')

ima = cv2.cvtColor(cv2.imread('Poema.bmp'),cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(ima,cmap='gray')

ima2 = ((ima)<255).astype(int) # Binarizar a menores de 255
# ---------------------------------- Corte Renglon --------------------------
sumaXFilas = np.sum(ima2,axis=1)
plt.figure()
plt.plot(sumaXFilas)

indicesXFilas = np.where(sumaXFilas!=0)
cadaRenglon = CortarImagen(indicesXFilas, 1)

corte_1 = indicesXFilas[0][cadaRenglon[0]]
corte_2 = indicesXFilas[0][cadaRenglon[0 + 1]]
        
renglonActual = ima2[corte_1:corte_2,:]
plt.figure()
plt.imshow(renglonActual)

# ---------------------------------- Corte Columna de renglon -----------
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
    
letraRenglon = [0]
palabraRenglon = [0]
indicesXColLetra = np.where(sumaXColumnasLetra!=0)
cadaLetra = CortarImagen(indicesXColLetra, 1)
    
P = []

dim = (45,36)

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
    P.append(vectorLetra)


P = np.array(P)
df = pd.DataFrame(P)
df.to_csv('P.csv')  

