# Le Résidu d'Erreur Binaire ($e_t$) : Définition, Mathématiques et Rôle Quantitatif

Ce document explique en détail la signification du terme **« résidu d'erreur binaire à chaque pas de temps $t$ »** utilisé dans la fonction `compute_directional_target_and_residuals` du fichier [`src/primary_model.py`](src/primary_model.py).

---

## 1. Décomposition Mot par Mot

Le terme se décompose en 4 notions fondamentales :

1. **Résidu** : En statistiques et modélisation, le résidu représente l'écart entre la valeur prédite par un modèle et la valeur réelle observée (ce qu'il « reste » d'inexpliqué par le modèle).
2. **Erreur** : Désigne l'événement où la prédiction du modèle ne correspond pas à la réalité du marché.
3. **Binaire** : Ne peut prendre que **deux états discrets** possibles :
   - **$0$** : Pas d'erreur (le modèle a vu juste).
   - **$1$** : Erreur (le modèle s'est trompé).
4. **À chaque pas de temps $t$** : Cette évaluation est calculée chronologiquement barre par barre (heure par heure dans notre série OHLCV 1h).

---

## 2. Formulation Mathématique en $\LaTeX$

Soient :
- $\hat{y}_t \in \{-1, +1\}$ : La prédiction directionnelle produite par le modèle primaire à l'instant $t$.
  - $\hat{y}_t = +1$ : Anticipation d'une hausse (Achat / Long).
  - $\hat{y}_t = -1$ : Anticipation d'une baisse (Vente / Short).
- $y_t = \text{sign}(r_{t+1}) \in \{-1, +1\}$ : La direction réelle constatée sur le marché à la période suivante $t+1$.

Le **résidu d'erreur binaire** à l'instant $t$, noté $e_t$, est formellement défini par :

$$e_t = \begin{cases} 0 & \text{si } \hat{y}_t = y_t \quad (\text{Prédiction exacte, aucun résidu d'erreur}) \\ 1 & \text{si } \hat{y}_t \neq y_t \quad (\text{Prédiction fausse, résidu d'erreur unitaire}) \end{cases}$$

### Notation avec la fonction indicatrice $\mathbb{I}$
De manière équivalente et compacte :

$$e_t = \mathbb{I}(\hat{y}_t \neq y_t)$$

### Forme algébrique
Puisque $\hat{y}_t \in \{-1, +1\}$ et $y_t \in \{-1, +1\}$, le produit $\hat{y}_t \cdot y_t$ vaut $+1$ en cas d'accord et $-1$ en cas de désaccord :

$$e_t = \frac{1 - \hat{y}_t \cdot y_t}{2}$$

---

## 3. Exemple Concret Heure par Heure

| Heure ($t$) | Signal du Modèle ($\hat{y}_t$) | Mouvement Réel ($y_t$) | Résidu d'Erreur ($e_t$) | Diagnostic |
| :---: | :---: | :---: | :---: | :--- |
| **10h00** | $+1$ (Achat) | $+1$ (Hausse) | **$0$** | Prédiction correcte $\implies$ Gain potentiel |
| **11h00** | $+1$ (Achat) | $-1$ (Baisse) | **$1$** | **Erreur du modèle** $\implies$ Perte évitée si filtrée |
| **12h00** | $-1$ (Vente) | $-1$ (Baisse) | **$0$** | Prédiction correcte $\implies$ Gain potentiel |
| **13h00** | $-1$ (Vente) | $+1$ (Hausse) | **$1$** | **Erreur du modèle** $\implies$ Perte évitée si filtrée |
| **$T$ (Dernière barre)** | $+1$ (Achat) | Inconnu (`NaN`) | **`NaN`** | Le futur $t+1$ n'a pas encore eu lieu |

---

## 4. Rôle Stratégique dans l'Architecture à Deux Étages (Méta-Labelling)

Dans les marchés financiers, aucun modèle primaire ne peut prédire l'avenir avec $100\%$ de certitude en raison de la non-stationnarité des régimes.

Le concept clé de votre architecture quantitative consiste à **ne pas forcer le modèle primaire à être parfait**, mais à **prédire ses moments de défaillance** :

```
                  ┌─────────────────────────────────────┐
                  │ Modèle Primaire (Chronos-Bolt)      │
                  │ Prédit la direction du marché : ŷ_t │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ Calcul du Résidu d'Erreur e_t in {0, 1}                │
        │ e_t = II(ŷ_t != y_t)                                   │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼ (Sert de Cible Y)
        ┌────────────────────────────────────────────────────────┐
        │ Arbre Substitut / Coupe-Circuit (Decision Tree)        │
        │ Prédit la probabilité d'erreur : P(e_t = 1 | X_t)      │
        └────────────────────────────┬───────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        Si P(e_t = 1) >= 52%                    Si P(e_t = 1) < 52%
        Coupe-circuit activé                    Trade autorisé
        S_t = 0 (100% Cash)                     S_t = 1 (Position active)
```

1. La série temporelle des $e_t$ devient la **variable cible** ($Y$) du second modèle (l'Arbre de Décision).
2. L'Arbre de Décision apprend à faire le lien entre les anomalies de marché $X_t = [d_{\mathcal{W}, t}, \, \sigma_{GK, t}]$ et l'apparition de $e_t = 1$.
3. Lorsque $\mathbb{P}(e_t = 1 \mid X_t) \ge \tau$, le coupe-circuit bascule le portefeuille à $100\%$ en liquidité (**Cash**), immunisant le capital contre les régimes toxiques.
