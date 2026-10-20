import numpy as np
from sklearn import datasets 
import sklearn

iris = datasets.load_iris()

X = iris.data
Y = iris.target

#metrics.pairwise.euclidean_distances 
#neighbors.KNeighborsClassifier

def PPV(X,Y):
    y_etiquette = []
    compteur_erreur = 0
    for i in range(len(X)):
        X_test = X[i].reshape(1,-1)

        X_train = np.delete(X,i,axis=0)
        Y_train = np.delete(Y,i)

        distances = sklearn.metrics.pairwise.euclidean_distances(X_train,X_test)

        voisin = np.argmin(distances)

        y_etiquette.append(Y_train[voisin])

    for k in range(len(y_etiquette)):
        if(y_etiquette[k] != Y[k]):
            compteur_erreur += 1

    pourcentage_erreur = (compteur_erreur/len(Y)) * 100

    return np.array(y_etiquette) , pourcentage_erreur

y,pourcentage_erreur = PPV(X,Y)
print(y)
print(f"{pourcentage_erreur}%")

def KNN():
    K_neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 3)
    K_neigh.fit(X, Y)


    y_pred_knn = K_neigh.predict(X)

    erreur_knn = (np.sum(y_pred_knn != Y) / len(Y)) * 100

    print("Prédictions Sklearn :", y_pred_knn)
    print(f"Erreur : {erreur_knn}%")

def CBN(X,Y):
    classe = np.unique(Y)
    proba_priori = []
    barycentre = []
    for c in classe:
        X_type_c = []
        for i in range(len(X)):
            if Y[i] == c :
                X_type_c.append(X[i])
        proba_priori.append(len(X_type_c) / len(X))
        barycentre.append(np.mean(X_type_c,axis=0))
    print(proba_priori)
    print(barycentre)

    CBN_res = []
    for x in X:
        res = []
        for k in range(len(classe)):
            proba_x_sachant_w = []

            for i in range(len(x)):
                xi = x[i]

                distances = []
                for bary in barycentre : 
                    distances.append(abs(xi - bary[i]))

                D = np.sum(distances)

                if D!=0 :
                    proba_x_sachant_w.append(1-(distances[k] / D))
                else :
                    proba_x_sachant_w.append(1.0 / len(classe))

            produit = np.prod(proba_x_sachant_w) * proba_priori[k]
            res.append(produit)

        maximum = np.argmax(res)
        CBN_res.append(maximum)

    return np.asarray(CBN_res)

l = CBN(X,Y)
print(l)