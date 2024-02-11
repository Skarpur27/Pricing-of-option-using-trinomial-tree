import numpy as np


class Node:
    """
    Classe représentant un noeud dans l'arbre trinomial.
    """

    def __init__(self, S, t, Tree=None):
        """
        Initialise un noeud avec ses paramètres.

        :param S: Prix du sous-jacent à ce noeud.
        :param t: Temps associé à ce noeud.
        :param Tree: Arbre auquel appartient ce noeud.
        """
        self.S = S
        self.t = t
        self.Tree = Tree
        self.mm_div = self.is_in_dividend_window()
        self.fwd_S = self.fwd_price()
        self.var = self.variance()
        self.Node_Up = None
        self.Node_Down = None
        self.Node_Next_Up = None
        self.Node_Next_Mid = None
        self.Node_Next_Down = None
        self.proba_cum = 0.0
        self.proba_faible = False
        self.pr_opt = None

    def is_in_dividend_window(self):
        """
        Vérifie si le noeud est dans la fenêtre du dividende.

        :return: True si le noeud est dans la fenêtre du dividende, sinon False.
        """
        if self.Tree is None:
            return False

        return self.t <= self.Tree.Mkt.date_div_a <= self.t + self.Tree.Mdl.delta_t_a

    def fwd_price(self):
        """
        Calcule et renvoie le prix forward du noeud.

        Si le noeud est dans la période de dividende (`mm_div` est True), le prix forward est ajusté
        pour prendre en compte le dividende. Si le prix forward devient négatif, une erreur est levée.

        :return: Prix forward du noeud, ajusté en fonction du dividende si applicable.
        """
        if self.Tree is None:
            return None
        elif not self.mm_div:
            # Calcul du prix forward sans ajustement de dividende
            return self.S * np.exp(self.Tree.Mkt.r * self.Tree.Mdl.delta_t_a)
        else:
            # Calcul du prix forward avec ajustement de dividende
            fwd_p = self.S * np.exp(self.Tree.Mkt.r * self.Tree.Mdl.delta_t_a) - self.Tree.Mkt.div
            if fwd_p > 0:
                return fwd_p
            else:
                raise ValueError("Le dividende est trop grand, entraînant un prix forward négatif.")

    def variance(self):
        """
        Calcule et renvoie la variance du prix du sous-jacent au noeud.

        La variance est calculée en prenant en compte le prix actuel du sous-jacent (S),
        le taux d'intérêt sans risque (r) et la volatilité du marché (vol).

        :return: La variance calculée du prix du sous-jacent.
        """
        if self.Tree is None:
            return None
        else:
            return (self.S ** 2) * np.exp(2 * self.Tree.Mkt.r * self.Tree.Mdl.delta_t_a) * \
                (np.exp((self.Tree.Mkt.vol ** 2) * self.Tree.Mdl.delta_t_a) - 1)

    def relations_haut_bas(self, node_up=None, node_down=None):
        """
        Établit les relations avec les noeuds supérieur et inférieur.

        Cette méthode relie le noeud actuel à un noeud supérieur et/ou inférieur si fournis,
        en établissant des liens bidirectionnels entre eux.

        :param node_up: Le noeud situé au-dessus du noeud actuel.
        :param node_down: Le noeud situé en dessous du noeud actuel.
        :return: Le noeud actuel avec ses nouvelles connexions.
        """
        # Connexion avec le noeud supérieur
        if node_up is not None:
            self.Node_Up = node_up
            node_up.Node_Down = self  # Connexion réciproque

        # Connexion avec le noeud inférieur
        if node_down is not None:
            self.Node_Down = node_down
            node_down.Node_Up = self  # Connexion réciproque

        return self

    def relations_suivantes(self, node_next_up=None, node_next_mid=None, node_next_down=None, pruning=False):
        """
        Établit des relations avec les noeuds suivants dans l'arbre.

        Cette méthode connecte le noeud actuel aux noeuds suivants : haut, milieu, et bas.
        Si le pruning est activé, certaines connexions peuvent être omises pour réduire la taille de l'arbre.

        :param node_next_up: Le noeud suivant situé en haut.
        :param node_next_mid: Le noeud suivant situé au milieu.
        :param node_next_down: Le noeud suivant situé en bas.
        :param pruning: Indique si le pruning est utilisé.
        :return: Le noeud actuel avec ses nouvelles connexions.
        """
        if not self.proba_faible and node_next_up is not None:
            self.Node_Next_Up = node_next_up

        if node_next_mid is not None:
            self.Node_Next_Mid = node_next_mid

        if not self.proba_faible and node_next_down is not None:
            self.Node_Next_Down = node_next_down
        return self

    def pup(self):
        """
        Calcule la probabilité d'une hausse du prix au prochain nœud.

        :return: La probabilité d'une hausse du prix.
        """
        # Calcul de la probabilité d'une hausse
        return (self.fwd_S / self.Node_Next_Mid.S - 1 - self.p_down * (1 / self.Tree.alpha - 1)) / \
            (self.Tree.alpha - 1)

    def pmid(self):
        """
        Calcule la probabilité d'un prix stable au prochain nœud.

        :return: La probabilité d'un prix stable.
        """
        # Calcul de la probabilité d'un prix stable
        return 1 - self.p_down - self.p_up

    def pdown(self):
        """
        Calcule la probabilité d'une baisse du prix au prochain nœud.

        :return: La probabilité d'une baisse du prix.
        """
        # Calcul de la probabilité d'une baisse
        return ((self.Node_Next_Mid.S ** -2) * (self.var + self.fwd_S ** 2) - 1 - (self.Tree.alpha + 1) *
                (self.Node_Next_Mid.S ** -1 * self.fwd_S - 1)) / \
            ((1 - self.Tree.alpha) * ((self.Tree.alpha ** -2) - 1))

    def proba_trans(self):
        """
        Calcule et assigne les probabilités de transition vers les noeuds suivants.

        Cette méthode prend en compte si le pruning est actif pour ajuster les probabilités.
        """
        if not self.proba_faible:
            self.p_down = self.pdown()
            self.p_up = self.pup()
            self.p_mid = 1 - self.p_up - self.p_down
        else:
            self.p_down = 0
            self.p_up = 0
            self.p_mid = 1

    def ajout_proba_cum(self, node_next_up, node_next_mid, node_next_down):
        """
        Ajoute les probabilités cumulées d'arriver aux 3 noeuds suivants à partir des probabilités du noeud actuel.
        """
        if not self.proba_faible:
            node_next_up.proba_cum += self.proba_cum * self.p_up
            node_next_mid.proba_cum += self.proba_cum * self.p_mid
            node_next_down.proba_cum += self.proba_cum * self.p_down
        else:
            node_next_mid.proba_cum += self.proba_cum * self.p_mid

    def proba_fbl(self):
        """
        Évalue si la probabilité cumulée du noeud est en dessous d'un seuil défini.

        Si la probabilité cumulée est inférieure au seuil défini dans le modèle,
        la méthode marque le noeud comme ayant une faible probabilité (`proba_faible`).
        Dans ce cas, les probabilités de transition sont recalculées pour ce noeud.

        :return: Un booléen indiquant si le noeud a une faible probabilité ou non.
        """
        if self.proba_cum < self.Tree.Mdl.seuil:
            self.proba_faible = True
            self.proba_trans()  # Recalcule les probabilités de transition pour le noeud
        else:
            self.proba_faible = False

        return self.proba_faible

    def next_mid(self, node_start):
        """
        Trouve le noeud le plus proche du prix forward actuel à partir d'un noeud de départ.
        :param node_start: Le noeud de départ pour la recherche.
        :return: Le noeud le plus proche du prix forward actuel.
        """
        noeud_actuel = node_start

        # Détermine la direction de la recherche (vers le haut ou vers le bas)
        direction = self.fwd_S >= node_start.S

        # Boucle de recherche du noeud le plus proche
        while True:
            if direction and self.is_close(noeud_actuel.Node_Down):
                return noeud_actuel.Node_Down

            elif not direction and self.is_close(noeud_actuel.Node_Up):
                return noeud_actuel.Node_Up

            # Crée un nouveau noeud si nécessaire et met à jour le noeud actuel
            if direction:
                noeud_actuel = self.get_or_create_next_node(noeud_actuel, up=True)
            else:
                noeud_actuel = self.get_or_create_next_node(noeud_actuel, up=False)

    def get_or_create_next_node(self, current_node, up=True):
        """
        Obtient ou crée le noeud suivant (en haut ou en bas) à partir du noeud actuel.
        :param current_node: Le noeud actuel.
        :param up: Booléen indiquant la direction (True pour haut, False pour bas).
        :return: Le noeud suivant.
        """
        next_node = current_node.Node_Up if up else current_node.Node_Down
        if next_node:
            return next_node

        # Crée un nouveau noeud si nécessaire
        new_price = current_node.S * self.Tree.alpha if up else current_node.S / self.Tree.alpha
        new_node = Node(new_price, current_node.t, self.Tree)
        new_node.relations_haut_bas(node_up=current_node if not up else None, node_down=current_node if up else None)
        return new_node

    def is_close(self, other_node):
        """
        Vérifie si le prix forward de ce noeud est plus proche du prix S du noeud donné
        que des prix des noeuds supérieur et inférieur.

        :param other_node: Le noeud à comparer.
        :return: True si le prix forward est plus proche, sinon False.
        """
        s1 = self.S * (1 + self.Tree.alpha) / 2
        s2 = self.S * (1 + (1 / self.Tree.alpha)) / 2

        return s2 < other_node.S < s1

    def payoff(self):
        """
        Calcule le payoff de l'option en fonction du type d'option (Call ou Put).

        :return: Le payoff de l'option.
        """
        # Si l'option est de type Call
        if self.Tree.Opt.IsCall:
            return max(self.S - self.Tree.Opt.K, 0)
        # Si l'option est de type Put
        else:
            return max(self.Tree.Opt.K - self.S, 0)

    def calculate_option_price(self):
        # Si le noeud est une feuille (fin de l'arbre), on utilise le payoff directement
        if self.Node_Next_Mid is None:
            self.pr_opt = self.payoff()

        else:
            if self.pr_opt is None:
                # Calcul de la valeur de l'option pour le mouvement vers le haut
                p_up_value = self.p_up * (self.Node_Next_Up.calculate_option_price() if self.Node_Next_Up else 0)

                # Calcul de la valeur de l'option pour le mouvement au milieu
                p_mid_value = self.p_mid * self.Node_Next_Mid.calculate_option_price()

                # Calcul de la valeur de l'option pour le mouvement vers le bas
                p_down_value = (self.p_down *
                                (self.Node_Next_Down.calculate_option_price() if self.Node_Next_Down else 0))

                # Actualisation des valeurs calculées pour obtenir la valeur de l'option au noeud actuel
                self.pr_opt = self.Tree.FA * (p_up_value + p_mid_value + p_down_value)

            # Si l'option est américaine, on compare la valeur de l'exercice immédiat à la valeur calculée
            if self.Tree.Opt.IsAmerican:
                self.pr_opt = max(self.pr_opt, self.payoff())

        return self.pr_opt
