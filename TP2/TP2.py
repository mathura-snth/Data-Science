import numpy as np
from sklearn.preprocessing import *
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

# A. Normalisation de données
X = [
    [1 , -1, 2],
    [2, 0, 0],
    [0, 1, -1]
]
X = np.array(X)
print(f"Matrice : \n{X}\n")

mean_value = np.mean(X)
print(f"Moyenne : {mean_value}\n")

var_value = np.var(X)
print(f"Variance : {var_value}\n")

X_scaled = scale(X)
print(f"Matrice Normalisé :\n{X_scaled}\n")
#Les valeurs sont centrées et reduites

mean_value_scaled = np.mean(X_scaled)
print(f"Moyenne de X normalisé : {mean_value_scaled}\n")

std_value_scaled = np.std(X_scaled)
print(f"Variance de X normalisé : {std_value_scaled}\n")

#La moyenne est proche de 0 et la variance vaut 1 donc
# cela valide le fait que notre matrice X a bien été normalisé


#B. Normalisation MinMax

X2 = [
    [1 , -1, 2],
    [2, 0, 0],
    [0, 1, -1]
]
X2 = np.array(X2)
print(f"Matrice X2 :\n {X2}\n")

mean_variables = np.mean(X2, axis=0)
print(f"Moyenne : {mean_variables}\n")

scaler = MinMaxScaler()
X2_scaled = scaler.fit_transform(X2)
print(f"Matrice X2 normalisé :\n {X2_scaled}\n")

mean_variables_scaled = np.mean(X2_scaled, axis=0)
print(f"Moyenne des variables de X2 normalisé :\n {mean_variables_scaled}")

# La moyenne ne vaut pas 0.
# Les valeurs de la matrice sont entre [0,1].

#C. Visualisation de données

iris = datasets.load_iris()

X = iris.data
y = iris.target
noms_variables = iris.feature_names

fig, axes = plt.subplots(4, 4, figsize=(15, 15))

for i in range(4):
    for j in range(4):
        axes[i, j].scatter(X[:, j], X[:, i], c=y, cmap='viridis')
        
        if i == 3:
            axes[i, j].set_xlabel(noms_variables[j])
        if j == 0:
            axes[i, j].set_ylabel(noms_variables[i])

plt.tight_layout()
plt.show()

# La meilleur visualisation est : 'petal length' x 'petal width' ou sa symetrie
# Raison : 
#   1. On a une meilleur visualiation qui permet de separer les classes plus facilement
#   2. On a une forte corrélation positive

#D. Reduction de dimensions et visualisation de données

pca = PCA()
IrisPCA  = pca.fit(iris.data).transform(iris.data)

lda = LDA()
IrisLDA = lda.fit(iris.data, iris.target).transform(iris.data)

plt.scatter(IrisPCA[:, 0], IrisPCA[:, 1], c=y, cmap='viridis')
plt.show()

plt.scatter(IrisLDA[:, 0], IrisLDA[:, 1], c=y, cmap='viridis')
plt.show()

#LDA etant supervisé elle se base sur l'etiquettage donc 
#on a la distance entre les points intra-classes qui diminuent et la 
#distance entre les points inter-classes qui augmentent, ce qui donne
#une meilleure visualisation