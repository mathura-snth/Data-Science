import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

nb_max = 1000
e = 0.01

def fonction_e(x) :
    return x**4 - 11*x**3 + 41*x**2 - 61*x + 30

def e_derive(x):
    return (4*x**3)-(33*x**2)+(82*x)-61

def dg(x0, n):
    list_xi = [x0]
    for i in range (1,nb_max) :
        xi_1 = list_xi[i-1] - n * e_derive(list_xi[i-1])
        list_xi.append(xi_1)
        if ((xi_1)-list_xi[i-1]) < e:
            break
    return list_xi,i

minimum = 10000
liste_x0 = [5,5,5,5,5,0]
liste_n = [0.001,0.01,0.1,0.17,1,0.001]
condition_index = 0
ite = 0
liste_des_minimums = []
for i in range(len(liste_x0)):
    val,ite = dg(liste_x0[i],liste_n[i])
    min_actuel = min(val)
    if min_actuel < minimum :
        liste_des_minimums.append(min_actuel)
        minimum = min_actuel
        condition_index = i

print(f"Le min est : {minimum} avec les param : dg({liste_x0[condition_index],liste_n[condition_index]}) avec {ite} iterations")
print(dg(liste_x0[condition_index],liste_n[condition_index]))

meilleurs_x, nb_iterations = dg(liste_x0[condition_index], liste_n[condition_index])

evolution_E = [fonction_e(x) for x in meilleurs_x]
liste_iterations = range(len(evolution_E))

plt.figure(figsize=(10, 6))
plt.plot(liste_iterations, evolution_E, marker='o', color='b', markersize=4, linestyle='-')
plt.title(f"Évolution de E(x) au fil des itérations\n(Paramètres : x0 = {liste_x0[condition_index]}, n = {liste_n[condition_index]})")
plt.xlabel("Numéro de l'itération")
plt.ylabel("Valeur de E(x)")
plt.grid(True)
plt.show()


import numpy as np

'''
1. Pour a : 2 * somme( xi * (axi + b - yi) )
   Pour b : 2 * somme( (axi + b - yi) )
'''

def dg_regression(x, y, eta, nombremax,e=1e-4):
    a, b = 0.0, 0.0 
    
    for _ in range(nombremax):
        dff = (a * x + b) - y
        
        grad_a = 2 * np.sum(x * dff)
        grad_b = 2 * np.sum(dff)
        
        a_suiv = a - eta * grad_a
        b_suiv = b - eta * grad_b

        a, b = a_suiv, b_suiv
    
    E_min = np.sum((a * x + b - y)**2)
    return a, b, E_min

x, y = datasets.make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
x = x.flatten()  

parametres = [
    (0.001, 100),
    (0.001, 500),
    (0.001, 1000),
    (0.01, 1000),
    (1, 1000)
]

print("Résultats de la Descente de Gradient :")
for eta, n_max in parametres:
    a_min, b_min, E_min = dg_regression(x, y, eta, n_max)
    print(f"eta={eta} | iter={n_max} | a={a_min} | b={b_min} | E={E_min}")

#Que constatez-vous ? 
#...\pythoncore-3.14-64\Lib\site-packages\numpy\_core\fromnumeric.py:83: RuntimeWarning: overflow encountered in reduce
#  return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
#...\tp3_desc_grad.py:65: RuntimeWarning: invalid value encountered in add
#  dff = (a * x + b) - y
#eta=1 | iter=1000 | a=nan | b=nan | E=nan

print("Question 4 : L'erreur E diminue lorsque le nombre d'itérations augmente de 100 à 1000. L'erreur 'NaN' obtenue avec eta=1 s'explique par le fait que le pas d'apprentissage est beaucoup trop grand par rapport à l'échelle des données, ce qui fait diverger les coefficients a et b")

from scipy.stats import linregress

result = linregress(x, y)
a_sci = result.slope
b_sci = result.intercept
E_sci = np.sum((a_sci * x + b_sci - y) ** 2)

print(f"a = {a_sci} | b = {b_sci} | E = {E_sci}")

print("Question 5 : Les deux méthodes aboutissent aux mêmes paramètres et au même coût minimal E: scipy résout le problème par une méthode analytique exacte instantanée, tandis que la descente de gradient s'en approche itérativement.")

a_dg, b_dg, e_min = dg_regression(x, y, eta=0.001, nombremax=1000)
x_droite = np.linspace(x.min(), x.max(), 100)
y_dg = a_dg * x_droite + b_dg
y_sci = a_sci * x_droite + b_sci

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', alpha=0.6, label="Données réelles")
plt.plot(x_droite, y_dg, color="red", linestyle="--", linewidth=3, label=f"DG (a={a_dg:.2f}, b={b_dg:.2f})")
plt.plot(x_droite, y_sci, color="green", linewidth=1.5, label=f"Scipy (a={a_sci:.2f}, b={b_sci:.2f})")

plt.title("Régression Linéaire : Descente de Gradient vs Scipy Linregress")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
# Puisque les deux méthodes aboutissent aux mêmes valeurs pour la pente $a$ et l'ordonnée à l'origine $b$, 
# les deux droites tracent exactement la même trajectoire et se superposent. 