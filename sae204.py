from ast import For
from statistics import linear_regression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap

#ouverture du csv
AdmissionsDF=pd.read_csv("df_admissions1.csv", sep=";")

#suppression des lignes contenant des valeurs manquantes
AdmissionsDF=AdmissionsDF.dropna()

#conversion en Array
AdmissionsArray=AdmissionsDF.to_numpy()

#Array
print(AdmissionsArray)

#DataFrame
print(AdmissionsDF)


def centrereduire(dataf):
    res=np.zeros(dataf.shape)
    moy=np.average(a=dataf,axis=0)
    std=np.std(a=dataf,axis=0)
    for i in range(len(dataf)):
        for j in range(dataf.shape[1]):
            res[i,j]=(dataf[i,j]-moy[j])/std[j]

    return res


AdmissionsArray_CR=centrereduire(AdmissionsArray)
print(AdmissionsArray_CR)
print(AdmissionsArray.shape)


Y=AdmissionsArray_CR[:,4]
X=AdmissionsArray_CR[:,[1,2,5,8]]

# print(X)
# print(Y)


linear_regression=LinearRegression()
linear_regression.fit(X,Y)
a=linear_regression.coef_
print(a)

print(linear_regression.score(X,Y))


MatriceCOV=np.cov(AdmissionsArray_CR,rowvar=False)

import numpy as np
import matplotlib.pyplot as plt

MatriceCOV = np.cov(AdmissionsArray_CR, rowvar=False)


# Définition d'une colormap personnalisée avec des dégradés de noir à rouge
colors = [(1, 1, 1), (1, 1, 0),(1, 0, 0),(0, 0, 0)]  # Noir à rouge
cmap_name = 'custom_red_black'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

# Créer une nouvelle figure
plt.figure()

# Afficher la matrice de covariance avec imshow et la colormap personnalisée
plt.imshow(MatriceCOV, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
plt.colorbar()

# Afficher les valeurs sur chaque case
for i in range(MatriceCOV.shape[0]):
    for j in range(MatriceCOV.shape[1]):
        plt.text(j, i, f'{MatriceCOV[i, j]:.2f}', ha='center', va='center', color='white')

plt.title('Matrice de covariance')

# Afficher la figure
plt.show()


print(MatriceCOV)



def find_max(dataf):
    max=0
    for i in range(len(dataf)):
        for j in range(dataf.shape[1]):
            #recherche d'un maximum inférieur à 0.84
            if(max<dataf[i,j] and dataf[i,j]<0.84):
                max=dataf[i,j]

    return max

print(find_max(MatriceCOV))



def coeff_corr_multiple(ypred):
    return 1-np.mean((ypred-Y)**2)/np.var(Y)

y_pred=X.dot(a)
print(coeff_corr_multiple(y_pred))
   


