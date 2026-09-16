import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_blobs

## Partie B
iris = datasets.load_iris()

data = iris.data
feature_names = iris.feature_names
target_names = iris.target_names
target = iris.target

print("Data : \n", data)
print("\nTargets : \n", target)
print("\nFeature Names : \n", feature_names)
print("\nTarget Names : \n", target_names)

print("\nClass name for each data :")
for i,classe in enumerate(target):
    print(target_names[classe])

# OU :   for i in range(len(feature_names)):
#           print(np.mean((data[:,i])))

print("\n\nmean : ", data.mean(0))
print("std : ", data.std(0))
print("min : ", data.min(0))
print("max : ", data.max(0))

print(f"\n\nNumber of data : {data.shape[0]}")
print(f"Number of features : {data.shape[1]}")
print(f"Number of classes : {target_names.size}")

## Partie C
mnist = datasets.fetch_openml('MNIST_784')
print(f"\n\nData :\n{mnist.data}\n")
print(f"Shape :\n{mnist.data.shape}\n")
print(f"Target :\n{mnist.target}\n")
print(f"Mean :\n{mnist.data.mean(0)}\n")
print(f"Std :\n{mnist.data.std(0)}\n")
print(f"Min :\n{mnist.data.min(0)}\n")
print(f"Max :\n{mnist.data.max(0)}\n")
print(f"Number of targets (unique) : {len(np.unique(mnist.target))}")

## Partie D
# help(make_blobs)
x,y = make_blobs(n_samples=1000, n_features=2, centers=4)

fig = plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Make blobs - 1000 données')
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.savefig('make_blobs_1000.png')

a,b = make_blobs(n_samples=100, n_features=2, centers=2)
c,d = make_blobs(n_samples=500, n_features=2, centers=3)

v = np.vstack((a,c))
h = np.hstack((b,d))

print("Affichage des trois ensembles")
fig2 = plt.figure(figsize=(15,5))

# affichage ensemble de 100 données
plt.subplot(1,3,1)
plt.scatter(a[:, 0], a[:, 1], c=b)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('Ensemble de 100 données')
plt.xlim(-15,15)
plt.ylim(-15,15)

# affichage ensemble de 500 données
plt.subplot(1,3,2)
plt.scatter(c[:, 0], c[:, 1], c=d)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('Ensemble de 500 données')
plt.xlim(-15,15)
plt.ylim(-15,15)

# affichage ensemble de la fusion
plt.subplot(1,3,3)
plt.scatter(v[:, 0], v[:, 1], c=h)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('Fusion des deux ensembles')
plt.xlim(-15,15)
plt.ylim(-15,15)

plt.savefig('fusion_ens.png')
plt.show()