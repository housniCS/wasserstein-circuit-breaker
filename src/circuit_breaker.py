"""
circuit_breaker.py
==================
Module de l'Étape 2 : Arbre Substitut Interprétable (Surrogate Tree)
utilisant la distance 1D de Wasserstein et la volatilité de Garman-Klass
comme indicateurs avancés d'invalidation du modèle primaire.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


class WassersteinCircuitBreaker:
    """
    Coupe-circuit statistique interprétable (White-Box Surrogate Model).

    Entraîne un arbre de décision contraint (max_depth=2, min_samples_leaf=50)
    sur les métriques d'instabilité de distribution (Wasserstein W_1) et de volatilité
    (Garman-Klass) pour prédire l'indicatrice d'erreur e_t in {0, 1} du modèle primaire.

    Si P(e_t = 1 | X_t) >= error_threshold (tau) :
        S_t = 0 (Coupe-circuit activé : 100% Cash / Désactivation des signaux)
    Sinon :
        S_t = 1 (Régime valide : Trade autorisé)
    """

    def __init__(
        self,
        max_depth: int = 2,
        min_samples_leaf: int = 50,
        error_threshold: float = 0.52,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Paramètres
        ----------
        max_depth : int
            Profondeur maximale de l'arbre (défaut = 2 pour garantir l'interprétabilité symbolique).
        min_samples_leaf : int
            Nombre minimum d'observations par feuille (évite le surapprentissage sur micro-régimes).
        error_threshold : float
            Seuil critique tau au-dessus duquel la position est coupée (ex. 0.52 = 52% de probabilité d'erreur).
        feature_names : list of str, optionnel
            Colonnes utilisées comme features (défaut: ["w1", "gk_vol"]).
        random_state : int
            Graine aléatoire pour la reproductibilité.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.error_threshold = error_threshold
        self.feature_names = feature_names or ["w1", "gk_vol"]
        self.random_state = random_state

        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.is_fitted: bool = False
        self.error_class_idx: int = 1  # Index correspondant à la classe e_t = 1

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "model_error"
    ) -> "WassersteinCircuitBreaker":
        """
        Ajuste l'arbre substitut sur l'historique fourni.

        Paramètres
        ----------
        df : pd.DataFrame
            DataFrame contenant les colonnes de features et la cible d'erreur.
        target_col : str
            Nom de la colonne contenant le résidu d'erreur e_t in {0, 1}.

        Retourne
        --------
        self : WassersteinCircuitBreaker
        """
        cols_needed = self.feature_names + [target_col]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le DataFrame pour l'entraînement : {missing}")

        # Nettoyage des valeurs manquantes (warm-up des rolling windows & dernière ligne)
        df_clean = df.dropna(subset=cols_needed).copy()

        if len(df_clean) < self.min_samples_leaf * 2:
            raise ValueError(
                f"Nombre d'échantillons valides insuffisant ({len(df_clean)}) "
                f"pour min_samples_leaf={self.min_samples_leaf}"
            )

        X = df_clean[self.feature_names].values
        y = df_clean[target_col].astype(int).values

        # Validation des classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(f"La cible {target_col} doit contenir au moins 2 classes (0 et 1), trouvé : {unique_classes}")

        self.tree.fit(X, y)
        self.is_fitted = True

        # Déterminer la position de la classe 1 dans classes_
        classes_list = list(self.tree.classes_)
        self.error_class_idx = classes_list.index(1)

        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule la probabilité conditionnelle d'erreur du modèle primaire :
        P(e_t = 1 | X_t).

        Retourne une pd.Series alignée sur l'index du DataFrame d'entrée.
        Les lignes ayant des NaN dans les features reçoivent NaN.
        """
        if not self.is_fitted:
            raise RuntimeError("L'arbre substitut doit être entraîné via .fit() avant d'effectuer des prédictions.")

        proba_series = pd.Series(np.nan, index=df.index, name="proba_error")
        valid_mask = df[self.feature_names].notna().all(axis=1)

        if valid_mask.any():
            X_valid = df.loc[valid_mask, self.feature_names].values
            probas = self.tree.predict_proba(X_valid)
            proba_series.loc[valid_mask] = probas[:, self.error_class_idx]

        return proba_series

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Génère le signal binaire du coupe-circuit : S_t in {0, 1}.

        S_t = 0 si P(e_t = 1 | X_t) >= error_threshold (Coupe-circuit activé / 100% Cash)
        S_t = 1 si P(e_t = 1 | X_t) <  error_threshold (Trade autorisé)

        Retourne une pd.Series alignée sur l'index de df (NaN où les features sont inconnues).
        """
        proba_err = self.predict_proba(df)
        signal = pd.Series(np.nan, index=df.index, name="circuit_breaker_signal")

        valid_mask = proba_err.notna()
        # S_t = 0 quand l'erreur anticipée est trop élevée (seuil dépassé), 1 sinon
        signal.loc[valid_mask] = np.where(
            proba_err.loc[valid_mask] >= self.error_threshold,
            0,
            1
        ).astype(int)

        return signal

    def explain_rules(self) -> str:
        """
        Exporte les règles de décision symboliques déduites par l'arbre sous forme de texte clair.
        """
        if not self.is_fitted:
            raise RuntimeError("L'arbre n'a pas encore été entraîné.")

        raw_tree_text = export_text(self.tree, feature_names=self.feature_names)
        
        explanation = [
            "=" * 70,
            " RÈGLES SYMBOLIQUES DU COUPE-CIRCUIT (SURROGATE DECISION TREE)",
            f" Features : {self.feature_names}",
            f" Seuil critique d'erreur (tau) : {self.error_threshold:.1%}",
            "=" * 70,
            "\nArbre de décision brut (sklearn) :",
            raw_tree_text,
            "-" * 70,
            "INTERPRÉTATION PRATIQUE DU COUPE-CIRCUIT :",
        ]

        # Analyse des feuilles
        n_nodes = self.tree.tree_.node_count
        children_left = self.tree.tree_.children_left
        children_right = self.tree.tree_.children_right
        value = self.tree.tree_.value
        n_node_samples = self.tree.tree_.n_node_samples

        leaf_count = 0
        for i in range(n_nodes):
            is_leaf = children_left[i] == children_right[i]  # True si feuille
            if is_leaf:
                leaf_count += 1
                counts = value[i][0]
                total = counts.sum()
                err_prob = counts[self.error_class_idx] / total if total > 0 else 0.0
                samples = int(n_node_samples[i])
                decision = "COUPE-CIRCUIT (CASH = 0)" if err_prob >= self.error_threshold else "TRADE AUTORISÉ (S = 1)"
                explanation.append(
                    f" - Feuille #{leaf_count} (n={samples} barres) : "
                    f"P(Erreur) = {err_prob:.1%} -> Décision : {decision}"
                )

        explanation.append("=" * 70)
        return "\n".join(explanation)


def apply_circuit_breaker(
    df: pd.DataFrame,
    breaker: WassersteinCircuitBreaker
) -> pd.DataFrame:
    """
    Fonction utilitaire appliquant le coupe-circuit sur un DataFrame :
    Ajoute les colonnes :
    - 'proba_error' : Probabilité d'échec du modèle primaire
    - 'circuit_breaker' : S_t in {0, 1} (0 = cash, 1 = trade)
    """
    df_out = df.copy()
    df_out["proba_error"] = breaker.predict_proba(df_out)
    df_out["circuit_breaker"] = breaker.predict(df_out)
    return df_out
