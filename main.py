from Classe import Model, Option, Market, Tree
from Fonction import (plot_trinomial_tree,plot_differences_and_times,plot_difference_vs_strike,pricing)
import sys
from datetime import date
import time

#Augmente le nombre de récursion possible
sys.setrecursionlimit(100000)

#Départ chronomètre
start_time = time.time()

# Définition de trois dates : date_dep, date_mat et date_div
date_dep = date(2023, 9, 1)
date_mat = date(2024, 7, 1)
date_div = date(2024, 3, 1)

# Caractéristiques de marché
taux_interet = 0.02
spot = 100
volatilite = 0.25
dividende = 0

# Caractéristiques de modèle
nb_pas = 200
seuil = 0
greeks = True
graph = True # Ne peut s'activer si plus de 8 pas

# Caractéristiques de l'option
strike = 101
option_am = False
option_call = True

# Initialisation d'un marché, d'un modèle et d'une option
marche = Market(taux_interet, spot, volatilite, dividende, date_div)
modele = Model(date_dep, date_mat, nb_pas, seuil, greeks,graph)
option = Option(strike, date_mat, option_am, option_call)

# Création de l'objet arbre + pricing
arbre = Tree(marche,option,modele)
pricing(arbre)

# Fonction permettant de sortir un graph sur la convergence de l'arbre trinominal et BS
#plot_differences_and_times(5000,marche, modele, option,15)

# Fonction permettant de sortir un graph sur la différence de prix entre l'arbre trinomial et BS en fonction du strike
#plot_difference_vs_strike(marche, modele, option,(50,170,0.1))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Temps écoulé: {elapsed_time} secondes")

