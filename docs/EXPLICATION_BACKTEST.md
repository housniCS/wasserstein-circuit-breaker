# Explication Détaillée : Moteur de Backtest Vectorisé Net de Frais

Ce document détaille les principes mathématiques, la mécanique algorithmique et les métriques financières du moteur de backtest vectorisé (`src/backtest.py`).

---

## 1. Qu'est-ce qu'un Backtest Vectorisé ?

Il existe deux manières d'évaluer une stratégie quantitative sur données historiques :

1. **Approche Événementielle (*Event-Driven*)** :
   - Simule barre après barre chaque ordre d'achat/vente à l'aide d'une boucle `for`.
   - Souvent lourd, lent et difficilement parallélisable.
2. **Approche Vectorisée (*Vectorized*)** :
   - Manipule la série temporelle entière sous forme de vecteurs (tableaux NumPy / séries Pandas).
   - Les opérations sur les prix, les positions et les frais sont calculées en bloc par des opérations matricielles $\mathcal{O}(N)$.
   - Exécution quasi instantanée, idéale pour tester et valider des stratégies systématiques.

---

## 2. La Règle d'Or Temporelle : Aucun Biais d'Anticipation (*No Lookahead Bias*)

En trading quantitatif, il est impératif d'aligner strictement les temporalités pour ne jamais utiliser une information future pour calculer un gain passé.

### Chronologie des événements :
1. À la clôture de la barre à l'instant **$t-1$**, on observe toutes les données passées et le modèle prend une décision de position : $\text{pos}_{t-1}$.
2. Durant l'intervalle entre $t-1$ et $t$, le marché évolue et produit le rendement $r_t$ :
   $$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
3. Le rendement brut généré par le portefeuille sur la période $[t-1, t]$ est :
   $$r_{\text{brut}, t} = \text{pos}_{t-1} \times r_t$$

> **Implémentation Pandas :**
> On décale la position d'un cran vers le bas avec `.shift(1)` afin de multiplier la décision prise en $t-1$ par le rendement constaté en $t$ :
> ```python
> df["strat_ret_gross"] = df["position"].shift(1) * df["ret"]
> ```

---

## 3. Les Trois Stratégies Comparées

| Stratégie | Formule de la Position à $t$ | Valeurs Possibles | Signification Économique |
| :--- | :--- | :--- | :--- |
| **1. Stratégie Brute** (sans filtre) | $\text{pos}_{\text{brut}, t} = \hat{y}_t$ | $\{-1, +1\}$ | Toujours investi (100% Long ou 100% Short) |
| **2. Stratégie Filtrée** (+ Coupe-circuit) | $\text{pos}_{\text{filt}, t} = \hat{y}_t \times S_t$ | $\{-1, 0, +1\}$ | Investi uniquement si $S_t = 1$. Si le coupe-circuit s'active ($S_t = 0$), basculement intégral en **100% Cash**. |
| **3. Benchmark (Buy & Hold)** | $\text{pos}_{\text{B\&H}, t} = 1$ | $\{+1\}$ | Conserver le Bitcoin passivement |

---

## 4. Modélisation Réaliste des Frais de Transaction (Frictions)

Les frais d'échange (*trading fees*) et le glissement de prix (*slippage*) peuvent transformer une stratégie théoriquement gagnante en stratégie perdante.

On applique un taux de friction réaliste de **$10\text{ bps}$ ($0.10\,\%$)** par rotation de portefeuille.

### A. Calcul du Turnover (Rotation)
Le changement de position entre deux pas de temps consécutifs vaut :

$$\Delta \text{pos}_t = |\text{pos}_t - \text{pos}_{t-1}|$$

* **Passage de Long ($+1$) à Short ($-1$)** :
  $$\Delta \text{pos}_t = |(-1) - (+1)| = 2 \implies \text{Frais} = 2 \times 0.10\% = 0.20\%$$
  *(Fermeture du long + ouverture du short)*
* **Passage de Long ($+1$) à Cash ($0$)** :
  $$\Delta \text{pos}_t = |0 - (+1)| = 1 \implies \text{Frais} = 1 \times 0.10\% = 0.10\%$$
* **Maintien de la position ($+1 \to +1$ ou $-1 \to -1$)** :
  $$\Delta \text{pos}_t = 0 \implies \text{Frais} = 0\%$$

### B. Formule des Frais et Rendement Net

$$\text{frais}_t = c \times |\text{pos}_t - \text{pos}_{t-1}| \quad \text{avec } c = 0.0010$$

Le rendement net réel de la stratégie à chaque heure $t$ est :

$$r_{\text{net}, t} = (\text{pos}_{t-1} \times r_t) - \text{frais}_t$$

---

## 5. Les Métriques Quantitatives Calculées

### 1. Courbe d'Équité (PnL Net Cumulé)
Capital cumulé normalisé (base 1.0 au départ) :

$$\text{Equity}_t = \prod_{i=1}^t (1 + r_{\text{net}, i})$$

### 2. Rendement Total Net
$$\text{Rendement Total} = \text{Equity}_T - 1$$

### 3. Sharpe Ratio Annualisé
Le Bitcoin cotant en continu 24h/24 et 365 jours par an, le nombre d'heures par an est $24 \times 365 = 8\,760$ :

$$\text{Sharpe} = \sqrt{8760} \times \frac{\mathbb{E}[r_{\text{net}}]}{\sigma(r_{\text{net}})}$$

### 4. Maximum Drawdown (MDD)
Mesure la perte maximale enregistrée depuis le sommet historique le plus élevé de la courbe de capital :

$$\text{Peak}_t = \max_{s \le t} \text{Equity}_s$$

$$\text{Drawdown}_t = \frac{\text{Peak}_t - \text{Equity}_t}{\text{Peak}_t}$$

$$\text{MDD} = \max_t (\text{Drawdown}_t)$$

### 5. Ratio de Calmar
Rapport entre le rendement annualisé et le Maximum Drawdown :

$$\text{Calmar} = \frac{\text{Rendement Annualisé}}{\text{MDD}}$$

### 6. Taux d'Inactivité (Cash Time)
Pourcentage du temps où la stratégie évite le risque de marché :

$$\% \text{ Cash} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(\text{pos}_t = 0)$$

---

## 6. Architecture du Code dans `src/backtest.py`

Le module fournit 3 fonctions modulaires :

1. `backtest_strategy(returns, positions, cost_bps=10.0)` :
   - Calcule les rendements bruts, les rotations de position, les frais et l'équité cumulée nette.
2. `compute_metrics(equity_series, net_returns)` :
   - Extrait le Sharpe, le MDD, le rendement total, la volatilité annualisée et le temps en cash.
3. `compare_strategies(df, cost_bps=10.0)` :
   - Exécute et compare les 3 stratégies :
     1. Modèle Primaire Brut
     2. Modèle Primaire + Wasserstein Circuit-Breaker
     3. Buy & Hold BTC
