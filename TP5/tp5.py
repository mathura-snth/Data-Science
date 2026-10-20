import numpy as np
from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


iris = datasets.load_iris()

def analyse(iris):
    nb_data = len(iris.data)
    nb_var = len(iris.feature_names)
    moyenne = np.mean(iris.data,axis=0)
    ecart = np.std(iris.data,axis=0)
    mini = np.min(iris.data,axis=0)
    maxi = np.max(iris.data,axis=0)

    pca = PCA()
    IrisPCA  = pca.fit(iris.data).transform(iris.data)

    lda = LDA()
    IrisLDA = lda.fit(iris.data, iris.target).transform(iris.data)

    plt.scatter(IrisPCA[:, 0], IrisPCA[:, 1], c=iris.target, cmap='viridis')
    plt.title("PCA")
    plt.show()

    plt.scatter(IrisLDA[:, 0], IrisLDA[:, 1], c=iris.target, cmap='viridis')
    plt.title("LDA")
    plt.show()


    return nb_data, nb_var, len(iris.target_names), moyenne, ecart, mini, maxi

res = analyse(iris)
print(res)

# interpretation + axes

def kmeans_manuel(X, k):    
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroides_liste = []

    for index in indices:
        point = X[index]
        centroides_liste.append(point)

    cluster = np.zeros(X.shape[0])

    for z in range(100):

        distances = np.zeros((X.shape[0], k))

        for i in range(X.shape[0]):
            point_actuel = X[i]

            for j in range(k):
                centroide_actuel = centroides_liste[j] 

                somme_des_carres = 0
                for d in range(X.shape[1]):
                    ecart = point_actuel[d] - centroide_actuel[d]
                    somme_des_carres += ecart ** 2

                distances[i][j] = somme_des_carres

        nouveaux_cluster = np.argmin(distances, axis=1)

        liste_nouveaux_centroides = []

        for c in range(k):
            if (nouveaux_cluster == c).any() :
                points = []
                for i in range(len(X)):
                    if nouveaux_cluster[i] == c:
                        points.append(X[i])
                nouveau_centre = np.array(np.mean(points, axis=0))

                liste_nouveaux_centroides.append(nouveau_centre)
            else :
                liste_nouveaux_centroides.append(centroides_liste[c]) 

        if np.array_equal(cluster, nouveaux_cluster):
            break

        cluster = nouveaux_cluster
        centroides_liste = liste_nouveaux_centroides
    return cluster

cluster = kmeans_manuel(iris.data, 3)
print(cluster)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=100).fit(iris.data)
nouveaux_cluster = kmeans.labels_
print(nouveaux_cluster)

## verifier pour différentes execution du meme code (car pris au hasard) + interpreter

for i in range(2,11):
    best = -1
    meilleur_k = -1
    meilleur_solution = -1
    for j in range(10):
        cluster = kmeans_manuel(iris.data, i)
        score = silhouette_score(iris.data, cluster)
   
        if score > best :
            best = score
            meilleur_k = i
            meilleur_solution = cluster
