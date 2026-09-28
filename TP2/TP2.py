import numpy as np
from sklearn.preprocessing import *
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

X = [
    [1 , -1, 2],
    [2, 0, 0],
    [0, 1, -1]
]

print(np.array(X))

mean_value = np.mean(X)
print(mean_value)
std_value = np.std(X)
print(std_value)

X_scaled = scale(X)
print(X_scaled)

mean_value_scaled = np.mean(X_scaled)
print(mean_value_scaled)

std_value_scaled = np.std(X_scaled)
print(std_value_scaled)

X2 = [
    [1 , -1, 2],
    [2, 0, 0],
    [0, 1, -1]
]

print(np.array(X2))

mean_variables = np.mean(X2, axis=0)
print(mean_variables)

scaler = MinMaxScaler()
X2_scaled = scaler.fit_transform(X2)
print(X2_scaled)

mean_variables_scaled = np.mean(X2_scaled, axis=0)
print(mean_variables_scaled)

# Partie C

iris = datasets.load_iris()

X = iris.data
y = iris.target
noms_variables = iris.feature_names

fig, axes = plt.subplots(4, 4, figsize=(15, 15))

for i in range(4):
    for j in range(4):
        axes[i, j].scatter(X[:, j], X[:, i], c=y, cmap='brg', edgecolor='k')
        
        if i == 3:
            axes[i, j].set_xlabel(noms_variables[j])
        if j == 0:
            axes[i, j].set_ylabel(noms_variables[i])

plt.tight_layout()
plt.show()

pca = PCA()
iris_pca = pca.fit(iris.data).transform(iris.data)

lda = LDA()
iris_lda = lda.fit(iris.data, iris.target).transform(iris.data)

plt.scatter(iris_pca[:, 0], iris_pca[:, 1], c=y, cmap='brg', edgecolor='k')

plt.show()


plt.scatter(iris_lda[:, 0], iris_lda[:, 1], c=y, cmap='brg', edgecolor='k')

plt.show()