from scipy.stats import norm
import numpy as np
from Node_ import Node


class Market:
    """
    Classe représentant les paramètres du marché.
    """
    def __init__(self, r, S0, vol, dividend, date_div):
        """
        Initialise les paramètres du marché.

        :param r: Taux d'intérêt sans risque.
        :param S0: Prix du sous-jacent à l'instant initial.
        :param vol: Volatilité du marché.
        :param dividend: Dividende du sous-jacent.
        :param date_div: Date du dividende.
        """
        self.r = r
        self.S0 = S0
        self.vol = vol
        self.div = dividend
        self.date_div = date_div
        self.date_div_a = None


class Model:
    """
    Classe représentant le modèle utilisé pour l'option.
    """
    def __init__(self, date_deb, date_matu, N, seuil=1e-5, display_greeks=False, graph=False):
        """
        Initialise les paramètres du modèle.

        :param date_deb: L'instant initial de création de l'option.
        :param date_matu: Date de maturité de l'option.
        :param N: Nombre de pas.
        :param seuil: Seuil pour le pruning.
        :param display_greeks : Afficher ou non les grecques de l'option
        :param graph : Afficher ou non un petit graphique

        """
        self.date_deb = date_deb
        self.N = N
        self.matu = date_matu
        self.delta_t_j = (date_matu - date_deb).days / N
        self.delta_t_a = self.delta_t_j / 365
        self.ecart_date = (date_matu - date_deb).days / 365
        self.display_greeks = display_greeks
        self.graph = graph
        self.seuil = seuil


class Option:
    """
    Classe représentant une option.
    """
    def __init__(self, K, mat, AmericanOpt=False, CallOpt=False):
        """
        Initialise les paramètres de l'option.

        :param K: Strike price.
        :param mat: Time to maturity.
        :param AmericanOpt: Si True, c'est une option américaine. Par défaut à False.
        :param CallOpt: Si True, c'est une option d'achat. Par défaut à False.
        """
        self.K = K
        self.mat = mat
        self.IsAmerican = AmericanOpt
        self.IsCall = CallOpt


class Tree:
    """
    Classe représentant l'arbre trinomial utilisé pour évaluer l'option.
    """
    def __init__(self, Market, Option, Model, Root=None):
        """
        Initialise l'arbre avec ses paramètres.

        :param Root: Noeud racine de l'arbre.
        :param Market: Paramètres du marché.
        :param Option: Paramètres de l'option.
        :param Model: Modèle utilisé pour l'option.
        """
        self.Root = Root
        self.Mkt = Market
        self.Opt = Option
        self.Mdl = Model

        # Vérification de la date de dividende
        if not (self.Mdl.date_deb <= self.Mkt.date_div <= self.Mdl.matu):
            raise ValueError("La date de dividende doit être entre la date de départ et la date de maturité.")

        self.FA = np.exp(-self.Mkt.r * self.Mdl.delta_t_a)
        self.alpha = np.exp(self.Mkt.vol * np.sqrt(3 * self.Mdl.delta_t_a))
        self.PriceTree = None
        self.PriceBS = None
        self.greeks = None

    def construction(self):
        """
        Construit l'arbre trinomial.
        """
        self.init_tree()
        noeud_actu = self.Root

        for i in range(self.Mdl.N):

            noeud_tronc = noeud_actu

            self.build_and_connect_nodes(noeud_actu)
            self.update_upper_nodes(noeud_tronc)
            self.update_lower_nodes(noeud_tronc)

            noeud_actu = noeud_tronc.Node_Next_Mid

        self.tree_price()
        self.black_scholes_price()


    def update_upper_nodes(self, start_node):
        """
        Parcourt les noeuds supérieurs au tronc et met à jour leurs liaisons.
        """
        noeud_actu = start_node
        while noeud_actu.Node_Up is not None:
            noeud_actu = noeud_actu.Node_Up
            noeud_actu.proba_faible = noeud_actu.proba_fbl()

            noeud_suivant_mid = noeud_actu.next_mid(noeud_actu.Node_Down.Node_Next_Mid)

            noeud_actu.relations_suivantes(noeud_suivant_mid.Node_Up, noeud_suivant_mid,
                                           noeud_suivant_mid.Node_Down)
            noeud_actu.proba_trans()
            noeud_actu.ajout_proba_cum(noeud_actu.Node_Next_Up, noeud_actu.Node_Next_Mid,
                                       noeud_actu.Node_Next_Down)


    def update_lower_nodes(self, start_node):
        """
        Parcourt les noeuds inférieurs au tronc et met à jour leurs liaisons réciproques.
        """
        noeud_actu = start_node
        while noeud_actu.Node_Down is not None:
            noeud_actu = noeud_actu.Node_Down
            noeud_actu.proba_faible = noeud_actu.proba_fbl()

            noeud_suivant_mid = noeud_actu.next_mid(noeud_actu.Node_Up.Node_Next_Mid)

            noeud_actu.relations_suivantes(noeud_suivant_mid.Node_Up, noeud_suivant_mid,
                                           noeud_suivant_mid.Node_Down)
            noeud_actu.proba_trans()
            noeud_actu.ajout_proba_cum(noeud_actu.Node_Next_Up, noeud_actu.Node_Next_Mid,
                                       noeud_actu.Node_Next_Down)

    def build_and_connect_nodes(self, noeud_actu):
        """
        Construit les noeuds suivants et établit les connexions, puis calcule les probabilités.

        :param noeud_actu: Le noeud actuel à partir duquel les noeuds suivants sont construits.
        """
        date_suiv = noeud_actu.t + self.Mdl.delta_t_a

        # Création des 3 noeuds suivants
        noeud_actu.Node_Next_Up = Node(noeud_actu.fwd_S * self.alpha, date_suiv, self)
        noeud_actu.Node_Next_Mid = Node(noeud_actu.fwd_S, date_suiv, self)
        noeud_actu.Node_Next_Down = Node(noeud_actu.fwd_S / self.alpha, date_suiv, self)

        # Connexions des noeuds suivants
        noeud_actu.relations_suivantes(noeud_actu.Node_Next_Up, noeud_actu.Node_Next_Mid, noeud_actu.Node_Next_Down)

        # Connexions haut-bas
        noeud_actu.Node_Next_Mid.relations_haut_bas(noeud_actu.Node_Next_Up, noeud_actu.Node_Next_Down)

        # Calcul des probabilités de transition et cumulatives
        noeud_actu.proba_trans()
        noeud_actu.ajout_proba_cum(noeud_actu.Node_Next_Up, noeud_actu.Node_Next_Mid, noeud_actu.Node_Next_Down)

    def init_tree(self):
        """
        Initialise l'arbre avec le noeud racine et les paramètres de base.
        """
        self.Mkt.date_div_a = (self.Mkt.date_div - self.Mdl.date_deb).days / 365.0
        self.Root = Node(self.Mkt.S0, 0)
        self.Root.Tree = self
        self.Root.mm_div = self.Root.is_in_dividend_window()
        self.Root.fwd_S = self.Root.fwd_price()
        self.Root.var = self.Root.variance()
        self.Root.proba_cum = 1

    def tree_price(self):
        """
        Calcule et retourne le prix de l'option basé sur l'arbre trinomial.

        Cette méthode utilise le noeud racine de l'arbre pour déclencher un calcul récursif
        du prix de l'option à travers l'arbre trinomial. Le résultat est stocké dans l'attribut
        `PriceTree` de la classe pour un accès ultérieur.

        :return: Le prix de l'option calculé à partir de l'arbre trinomial.
        """
        self.PriceTree = self.Root.calculate_option_price()
        return self.PriceTree

    def black_scholes_price(self):
        """
        Calcule le prix de l'option selon la formule de Black-Scholes.

        :return: Prix de l'option selon Black-Scholes.
        """
        S = self.Mkt.S0
        K = self.Opt.K
        r = self.Mkt.r
        sigma = self.Mkt.vol

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * self.Mdl.ecart_date) / (sigma * np.sqrt(self.Mdl.ecart_date))
        d2 = d1 - sigma * np.sqrt(self.Mdl.ecart_date)

        if self.Opt.IsCall: # Call
            self.PriceBS = S * norm.cdf(d1) - K * np.exp(-r * self.Mdl.ecart_date) * norm.cdf(d2)
        else:  # Put
            self.PriceBS = K * np.exp(-r * self.Mdl.ecart_date) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return self.PriceBS

    def calculate_greeks(self, epsilon_price=5, epsilon_vol=0.01, epsilon_rate=0.01, epsilon_time=1):
        """
        Calcule les grecques de l'option.
        """
        self.delta, self.gamma = self.calculate_delta_gamma(epsilon_price)
        self.vega = self.calculate_vega(epsilon_vol)
        self.rho = self.calculate_rho(epsilon_rate)
        self.theta = self.calculate_theta(epsilon_time)

        return {
            'Delta': self.delta,
            'Gamma': self.gamma,
            'Vega': self.vega,
            'Rho': self.rho,
            'Theta': self.theta
        }

    def calculate_delta_gamma(self, epsilon_price):
        """
        Calcule et retourne à la fois le Delta et le Gamma de l'option.

        Delta est la variation du prix de l'option par rapport à une petite variation du prix de l'actif sous-jacent.
        Gamma est la variation du Delta par rapport à une petite variation du prix de l'actif sous-jacent.

        :param epsilon_price: Petite variation appliquée au prix de l'actif sous-jacent pour le calcul.
        :return: Un tuple contenant à la fois le Delta et le Gamma de l'option.
        """
        # Prix original du sous-jacent
        original_S = self.Mkt.S0

        # Calcul du prix de l'option avec une augmentation de epsilon_price
        self.Mkt.S0 = original_S + epsilon_price
        self.construction()
        price_up = self.tree_price()

        # Calcul du prix de l'option avec une diminution de epsilon_price
        self.Mkt.S0 = original_S - epsilon_price
        self.construction()
        price_down = self.tree_price()

        # Prix de l'option avec le prix du sous-jacent original
        self.Mkt.S0 = original_S
        self.construction()
        price_original = self.tree_price()

        # Calcul de Delta et Gamma en utilisant les prix calculés
        delta = (price_up - price_down) / (2 * epsilon_price)
        gamma = (price_up - 2 * price_original + price_down) / (epsilon_price ** 2)

        # Restauration du prix original du sous-jacent
        self.Mkt.S0 = original_S

        return delta, gamma

    def calculate_vega(self, epsilon_vol):
        """
        Calcule et retourne le Vega de l'option.
        """
        original_vol = self.Mkt.vol
        self.Mkt.vol += epsilon_vol
        self.construction()
        price_up_vol = self.tree_price()

        self.Mkt.vol = original_vol
        self.construction()
        price_original = self.tree_price()

        vega = (price_up_vol - price_original) / epsilon_vol
        self.Mkt.vol = original_vol  # Restaure la volatilité originale après le calcul
        return vega

    def calculate_theta(self, epsilon_time):
        """
        Calcule et retourne le Theta de l'option.
        """
        original_time = self.Mdl.ecart_date
        self.Mdl.ecart_date += epsilon_time / 365  # Convertit le temps en années
        self.construction()
        price_up_time = self.tree_price()

        self.Mdl.ecart_date = original_time
        self.construction()
        price_original = self.tree_price()

        theta = (price_original - price_up_time) / epsilon_time
        self.Mdl.ecart_date = original_time  # Restaure l'écart de temps original après le calcul
        return theta

    def calculate_rho(self, epsilon_rate):
        """
        Calcule et retourne le Rho de l'option.
        """
        original_r = self.Mkt.r
        self.Mkt.r += epsilon_rate
        self.construction()
        price_up_r = self.tree_price()

        self.Mkt.r = original_r
        self.construction()
        price_original = self.tree_price()

        rho = (price_up_r - price_original) / epsilon_rate
        self.Mkt.r = original_r  # Restaure le taux d'intérêt original après le calcul
        return rho

