import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut,cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_mutual_info_score, adjusted_rand_score,classification_report

#Question A.1 : 
def analyse(dateset,X,Y):
    nb_data = len(X)
    nb_var = np.shape(X)[1]
    nb_classe = len(np.unique(Y))
    moyenne = np.mean(X,axis=0)
    ecart = np.std(X,axis=0)
    mini = np.min(X,axis=0)
    maxi = np.max(X,axis=0)

    pca = PCA()
    XPCA  = pca.fit(X).transform(X)

    lda = LDA()
    XLDA = lda.fit(X, Y).transform(X)

    plt.scatter(XPCA[:, 0], XPCA[:, 1], c=Y, cmap='viridis')
    plt.title(f"PCA {dateset}")
    plt.show()

    plt.scatter(XLDA[:, 0], XLDA[:, 1], c=Y, cmap='viridis')
    plt.title(f"LDA {dateset}")
    plt.show()
    
    #Question C.1 : Clustering_donnes_diff_algo(X,Y)

    #Question D.1 : LeaveOneOut(X,Y)
    return nb_data, nb_var,nb_classe, moyenne, ecart, mini, maxi

#Question B.1 :
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


iris = datasets.load_iris()
X,Y = iris.data,iris.target
print(analyse("IRIS",X,Y))

#Question A.2 :
# L'ACP est une méthode non supervisée, elle projette les données dans un espace de dimension plus petite en cherchant à maximiser la variance.
# L'ADL est une méthode supervisée, on projette aussi mais cette fois en cherchant des axes optimaux en maximisant la séparabilité entre les classes

#Question B.1 (Suite) : 
print("1er lancement du KMeans Manuel pour k  = 3 :")
print(kmeans_manuel(X, 3))

print("Lancement du KMeans Sklearn pour k = 3")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=100).fit(X)
nouveaux_cluster = kmeans.labels_
print(nouveaux_cluster)
#Interpretation : Les deux algo sont differents. Je pense que c'est dû au choix aleatoire des centres

#Question B.2 :
print("2eme lancement du KMeans Manuel pour k  = 3 :")
print(kmeans_manuel(X, 3))
#Interpretation : On remarque bien que les deux lancement nous renvoie des resultats differents
#Celui est du au choix aleatoire de nos centres

#Question B.3 : 
best = -1
meilleur_k = -1
meilleur_solution = -1
for i in range(2,11):
    for j in range(10):
        cluster = kmeans_manuel(X, i)
        score = silhouette_score(X, cluster)
   
        if score > best :
            best = score
            meilleur_k = i
            meilleur_solution = cluster

print(f"Le nombre de cluster retenu est de {meilleur_k} avec score : {best}")
#Interpretation : La classe Y à 3 classe mais ici l'algo nous renvoie 2
#Une classe est hyper loin des autres, mais les 2 autres se chevauchent
#Donc je pense que Silhouette propose une separation plus simple

#Question B.4 :

pca = PCA()
iris_pca = pca.fit_transform(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

scatter1 = ax1.scatter(iris_pca[:, 0], iris_pca[:, 1], c=Y, cmap='viridis')
ax1.set_title("PCA")

scatter2 = ax2.scatter(iris_pca[:, 0], iris_pca[:, 1], c=meilleur_solution, cmap='viridis')
ax2.set_title("Meilleur Cluster")

plt.tight_layout()
plt.show()

#Interpretation : On a une meilleur visualisation de la sortie de score
#On a 2 classe au lieu de 3. La classe violet est bien separé dans les deux cas
#La classe bleau se chevauchent avec la classe jaune donc l'algo considere en une clase
#Donc les groupes obtenues sont coherant avec leur algo respective mais pas en terme de logique

#Question C.1 :
def Clustering_donnes_diff_algo(X,Y):

    print("\nLes etiquettes proposés par les 5 algos differents :\n")
    print("#" * 70)

    print("1er algo : KMeans Manuel")
    cluster_kmeans =  kmeans_manuel(X, 3)
    print(cluster_kmeans)
    score_interne_externe(X,Y,cluster_kmeans)

    print("2eme algo : MeanShift")
    clustering = MeanShift().fit(X)
    cluster_MeanShift = clustering.labels_
    print(cluster_MeanShift)
    score_interne_externe(X,Y,cluster_MeanShift)

    print("3eme algo : Affinity Propgation")
    clustering = AffinityPropagation(random_state=5).fit(X)
    cluster_AffinityPropagation = clustering.labels_
    print(cluster_AffinityPropagation)
    score_interne_externe(X,Y,cluster_AffinityPropagation)
    
    print("4eme algo : SpectralClustering")
    clustering = SpectralClustering(n_clusters=3,random_state=5).fit(X,Y)
    cluster_SpectralClustering = clustering.labels_
    print(cluster_SpectralClustering)
    score_interne_externe(X,Y,cluster_SpectralClustering)
    
    print("5eme algo : DBSCAN")
    clustering = DBSCAN().fit(X)
    cluster_DBSCAN = clustering.labels_
    print(cluster_DBSCAN)
    score_interne_externe(X,Y,cluster_DBSCAN)

    return 0

#Question C.2 :
def score_interne_externe(X,Y,cluster):
    davies = davies_bouldin_score(X, cluster)
    silhouette = silhouette_score(X, cluster)
    mutual_info = adjusted_mutual_info_score(Y,cluster)
    rand_score = adjusted_rand_score(Y,cluster)

    print("\nScores Internes : \n")

    print(f"DAVIES BOULDIN SCORE : {davies}")
    print(f"SILHOUETTE SCORE : {silhouette}")

    print("\nScores Externe : \n")

    print(f"ADJ MUTUAL INFO : {mutual_info}")
    print(f"ADJ RAND SCORE : {rand_score}\n")

    print("#" * 70)

Clustering_donnes_diff_algo(X,Y)

# La différence entre un score interne et un score externe repose sur les données requises et l'utilité du score :
# pour un score interne :
#       * on a seulement besoin des caractéristiques des données (donc X) et les clusters prédits, 
#       * on cherche à minimiser la distance intra cluster et maximiser la distance inter cluster

# pour un score externe :
#       * on a, en plus, besoin des vraies étiquettes (Y)
#       * le but est de vérifier le bon étiquettage (maximiser la similarité avec la répartition réelle)


# Question C.3 : 
# Compte tenu des résultats des différentes méthodes de clustering,
# les méthodes qui ont les meilleurs scores et dont les scores sont les plus homogènes sont : Spectral et Kmeans manuel

#Question D.1 :
def KNN(X,Y,loo):

    K_neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 1)

    y_pred_knn = cross_val_predict(K_neigh, X, Y, cv=loo)

    erreur_knn = (np.sum(y_pred_knn != Y) / len(Y)) * 100

    return y_pred_knn, erreur_knn

def gaussiannb(X,Y,loo) :
    gaussiannb = GaussianNB()

    y_pred_gaussiannb = cross_val_predict(gaussiannb, X, Y, cv=loo)

    err_gaussiannb = (np.sum(y_pred_gaussiannb != Y) / len(Y)) * 100

    return y_pred_gaussiannb, err_gaussiannb

def LeaveOneOut(X,Y):
    Clustering_donnes_diff_algo(X,Y)

    loo = LeaveOneOut()
    kNN_predict, erreur_knn = KNN(X,Y,loo)
    calcul("KNN", erreur_knn,Y,kNN_predict)

    gaussian_predict, erreur_guassian = gaussiannb(X,Y,loo)
    calcul("Gaussian",erreur_guassian,Y,gaussian_predict)

    MLP_predict =  cross_val_predict(MLPClassifier(max_iter=1000), X, Y, cv=loo)
    erreur_mlp = (np.sum(MLP_predict != Y) / len(Y)) * 100
    calcul("MLP", erreur_mlp,Y,MLP_predict)

    
    SVC_predict = cross_val_predict(SVC(), X, Y, cv=loo)
    erreur_svc = (np.sum(SVC_predict != Y) / len(Y)) * 100
    calcul("SVC", erreur_svc,Y,SVC_predict)

    return

#Question D.2 : 
def calcul(method, erreur,Y,predict):
    print(f"{method}\n")
    print(f"\nPourcentage d'Erreur : {erreur}\n")
    print("Rapport de classification :\n")
    print(classification_report(Y, predict))
    print("\n")
    print("#" * 70)

#Question D.3 : Comme pour la C.3 faut voir comment marche ces fonctions
#Ici j'avais utiliser Gemini parce que j'avais pas compris la consigne et me diriger vers un mur :lol:
#Faudra bien revoir ça ducoup

#Question E.1 : 
#On lance que sur ça donc faudra commentez les resultats

#Question E.2 : 
wine = datasets.load_wine()
X,Y = wine.data, wine.target
print(analyse("WINE X non centré & reduit",X,Y))

#Question E.3
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(analyse("WINE : X centré & reduit",X_scaled,Y))

#Interpretation : A faire (Je suppose que ça doit etre mieux
#On a centre et reduit donc les données sont au meme echelle)