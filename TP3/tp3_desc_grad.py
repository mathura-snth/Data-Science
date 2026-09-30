import numpy as np
import matplotlib.pyplot as plt

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


