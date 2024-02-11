import networkx as nx
import math

from matplotlib import patches

from Classe import Model, Option, Market, Tree, Node
import matplotlib.pyplot as plt
import numpy as np
import gc
import time
from Node_ import Node

"""
def plot_trinomial_tree(node, graph=None, pos=None, level=0, time=0):
    if graph is None:
        graph = nx.Graph()
        pos = {}

    node_name = f"{node.S:.1f}"  # Format simplifié pour les étiquettes
    graph.add_node(node_name)  # Ajouter le nœud actuel sans étiquette
    pos[node_name] = (time, -level)  # Position du nœud

    # Tracer des liens vers les enfants
    if node.Node_Next_Up:
        plot_trinomial_tree(node.Node_Next_Up, graph, pos, level+1, time+1)
    if node.Node_Next_Mid:
        plot_trinomial_tree(node.Node_Next_Mid, graph, pos, level, time+1)
    if node.Node_Next_Down:
        plot_trinomial_tree(node.Node_Next_Down, graph, pos, level-1, time+1)

    # Dessiner l'arbre
    if level == 0 and time == 0:
        nx.draw(graph, pos, with_labels=False, node_size=0, node_color="none", font_size=8)

        # Dessiner les lignes de connexion en premier
        edge_lines = nx.draw_networkx_edges(graph, pos)

        # Ajouter les étiquettes des prix dans des rectangles décalés
        for node, (x, y) in pos.items():
            plt.text(x, y, node, fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle="square,pad=0.2", edgecolor='black', facecolor='white'))

        plt.gca().invert_yaxis()  # Inversion de l'axe des y pour que le haut soit en haut
        plt.title("Visualisation de l'Arbre Trinomial")
        plt.xlabel("Temps")
        plt.ylabel("Niveau")
        plt.show()
"""



def plot_trinomial_tree(node, x=0, y=0, level=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.invert_yaxis()  # Pour que le nœud racine soit en haut

    # Affiche le prix du nœud actuel
    ax.text(x, y, f'{node.S:.4f}', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))

    # Si le nœud a un Node_Next_Up, dessinez une ligne vers lui et récursivement plottez cet arbre
    if node.Node_Next_Up:
        ax.plot([x, x + 1], [y, y - 1], 'k-')
        plot_trinomial_tree(node.Node_Next_Up, x + 1, y - 1, level + 1, ax)

    # Si le nœud a un Node_Next_Mid, dessinez une ligne vers lui et récursivement plottez cet arbre
    if node.Node_Next_Mid:
        ax.plot([x, x + 1], [y, y], 'k-')
        plot_trinomial_tree(node.Node_Next_Mid, x + 1, y, level + 1, ax)

    # Si le nœud a un Node_Next_Down, dessinez une ligne vers lui et récursivement plottez cet arbre
    if node.Node_Next_Down:
        ax.plot([x, x + 1], [y, y + 1], 'k-')
        plot_trinomial_tree(node.Node_Next_Down, x + 1, y + 1, level + 1, ax)

    if level == 0:  # Si c'est le nœud racine, affichez le graphique
        plt.show()


def generate_log_space(start, end, points_per_decade):
    current_exponent = math.floor(math.log10(start))
    end_exponent = math.ceil(math.log10(end))

    values = []

    while current_exponent <= end_exponent:
        base_value = 10 ** current_exponent
        for i in range(points_per_decade):
            value = base_value * (10 ** (i / points_per_decade))
            if value >= start and value <= end and value not in values:
                values.append(value)
        current_exponent += 1

    return values


def compute_differences_and_times_for_range(max_steps, marche, modele, option, points_per_decade, min_steps=1):
    differences = []
    execution_times = []
    steps_values = []

    steps_list = generate_log_space(min_steps, max_steps, points_per_decade)

    for NbSteps in steps_list:
        NbSteps = int(NbSteps)  # Convertir en entier
        modele_temp = Model(modele.date_deb, modele.matu, NbSteps, modele.seuil)

        start_time = time.time()  # Enregistrez le temps de départ

        # Création de l'arbre avec les nouveaux paramètres
        arbre = Tree(marche, option, modele_temp)
        arbre.construction()  # Construction de l'arbre
        pricing_arbre = arbre.tree_price()  # Prix de l'option selon l'arbre

        end_time = time.time()  # Enregistrez le temps de fin

        pricing_BS = arbre.black_scholes_price()  # Prix de l'option selon Black-Scholes
        differences.append((pricing_arbre - pricing_BS) * NbSteps)
        execution_times.append(end_time - start_time)
        steps_values.append(NbSteps)

        # Supprimer l'arbre et forcer la libération de la mémoire
        del arbre
        gc.collect()

    return steps_values, differences, execution_times


def plot_differences_and_times(max_steps, marche, modele, option, points_per_decade=10, min_steps=1):
    # Générer les données
    steps_values, differences, execution_times = compute_differences_and_times_for_range(max_steps, marche, modele,
                                                                                         option, points_per_decade,
                                                                                         min_steps)

    # Tracer la différence entre les prix de l'arbre et de Black-Scholes
    plt.figure(figsize=(10, 6))
    plt.plot(steps_values, differences, '-o', markersize=4, label='Difference (Tree - BS) x NbSteps')
    plt.xscale('log')
    plt.xlabel('NbSteps (log scale)')
    plt.ylabel('(Tree - BS) x NbSteps')
    plt.title('Difference between Tree and BS prices multiplied by NbSteps')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Tracer les temps d'exécution
    plt.figure(figsize=(10, 6))
    plt.plot(steps_values, execution_times, '-o', markersize=4, color='red', label='Execution Time')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('NbSteps (log scale)')
    plt.ylabel('Execution Time (seconds, log scale)')
    plt.title('Execution Time vs. NbSteps')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_difference_vs_strike(marche, modele, option, strike_range=(50, 150, 5)):
    """
    Trace la différence entre le prix de l'arbre trinomial et le prix de Black-Scholes en fonction du strike.

    :param marche: Objet représentant le marché.
    :param modele: Objet représentant le modèle.
    :param option: Objet représentant l'option.
    :param strike_range: Tuple contenant (strike_min, strike_max, intervalle).
    """

    strikes = np.arange(*strike_range)
    differences = []

    for K in strikes:
        option.K = K
        arbre = Tree(marche, option, modele)
        arbre.construction()
        differences.append(arbre.PriceTree - arbre.PriceBS)

        # Suppression des objets et forcer le garbage collection
        del arbre
        gc.collect()

    # Tracer le graphique
    plt.plot(strikes, differences, marker='o', linestyle='-')
    plt.title('Différence entre le prix de l\'arbre et Black-Scholes')
    plt.xlabel('Strike')
    plt.ylabel('Différence de prix')
    plt.grid(True)
    plt.show()


def pricing(arb):
    arb.construction()
    print(f'Prix arbre: {arb.PriceTree}')

    if not arb.Opt.IsAmerican and arb.Mkt.div == 0:
        print(f'Prix BS: {arb.PriceBS}')
        print(f'Ecart prix : {abs(arb.PriceTree - arb.PriceBS)}')

    if arb.Mdl.display_greeks:
        print(arb.calculate_greeks())

    if arb.Mdl.graph and arb.Mdl.N<=8:
        plot_trinomial_tree(arb.Root)