# Guide Complet et Architecture Globale du Projet

## Système de Trading Systématique avec Coupe-Circuit de Wasserstein (Distribution-Shift Resilience)

Ce document offre une vision intégrale et détaillée de l'ensemble du projet : sa thèse quantitative, son architecture à deux étages, ses fondations mathématiques, le rôle précis de chaque fichier source et les instructions d'exécution.

---

## 1. Thèse Quantitative et Vision Fondamentale

### Le Problème : La Non-Stationnarité et les Faux Signaux
En finance quantitative et sur les marchés de crypto-actifs, les séries temporelles sont **intrinsèquement non-stationnaires**. Les propriétés statistiques des cours (volatilité, asymétrie, kurtosis, dynamique du carnet d'ordres) changent brutalement selon les régimes de marché (phases de bull run haussier, dérives latérales, cascades de liquidation et flash-crashes).

Lorsqu'un modèle d'intelligence artificielle ou de Deep Learning (tel que les transformers de fondation de séries temporelles) tente de prédire la direction du marché en continu :
1. Dans les régimes normaux, il affiche une rentabilité acceptable.
2. Dans les régimes de crise ou d'anomalie de distribution (*out-of-distribution*), son ratio signal/bruit s'effondre.
3. Il génère alors des signaux directionnels erratiques qui entraînent des pertes massives et un turnover excessif, dévoré par les frais de transaction.

### La Solution : L'Approche à Deux Étages (*Meta-Labeling* & Coupe-Circuit)
Plutôt que de chercher un hypothétique « modèle parfait » impossible à concevoir, ce projet implémente une **stratégie de défense asymétrique du capital** :

1. **Étage 1 (Directionnel)** : Un modèle primaire prédit le sens probable du cours ($\hat{y}_t \in \{-1, +1\}$).
2. **Étage 2 (Contrôle du Risque)** : Un modèle substitut interprétable (arbre de décision CART) surveille en permanence la physique du marché (volatilité et divergence de distribution). Dès qu'il identifie une configuration statistique où le modèle primaire échoue fréquemment, il déclenche un **coupe-circuit déterministe** :
   
   $$\text{Position Finale} = \text{Signal Primaire} \times \text{Coupe-Circuit} \implies S_t = 0 \implies \mathbf{100\% \text{ Cash}}$$

Cette approche permet de préserver le capital durant les tempêtes de marché, diminuant drastiquement le *Maximum Drawdown* net de frais.

---

## 2. L'Architecture en 3 Étapes Clés

Le projet est structuré selon un pipeline séquentiel rigoureux en 3 étapes :

```
┌────────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1 : Données, Features & Modèles Primaires                        │
│ - Téléchargement/chargement OHLCV 1h (Binance BTC/USDT)                │
│ - Extraction vectorisée : Rendements r_t, Volatilité sigma_GK,         │
│   Distance de Wasserstein glissante W_1                                │
│ - Modèle Primaire : Baseline Momentum ou Deep Learning Chronos-Bolt    │
│ - Génération des résidus d'erreurs binaires : e_t = II(y_hat_t != y_t) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2 : Arbre Substitut Interprétable (Coupe-Circuit CART)           │
│ - Entraînement de l'arbre CART sur les résidus d'erreurs e_t           │
│ - Variables explicatives : X_t = [W_{1, t}, sigma_{GK, t}]             │
│ - Extraction automatique de règles symboliques lisibles 'if/then'      │
│ - Génération du signal de coupure : S_t in {0, 1}                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 3 : Backtest Vectorisé Institutionnel Net de Frais               │
│ - Absence de biais d'anticipation : r_{gross, t} = pos_{t-1} * r_t     │
│ - Frictions réalistes : 10 bps par changement de position              │
│ - Comparaison des 3 courbes : Brute vs Filtrée (+CB) vs Buy & Hold     │
│ - Tableau de bord : Sharpe, Calmar, Max Drawdown, Win Rate, Turnover   │
│ - Génération du rapport visuel haute résolution : results.png          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Cartographie Complète des Fichiers du Projet

```text
projet/
├── data/
│   ├── btc_1h.csv                     # Cache local des chandeliers 1h BTC/USDT
│   └── chronos_preds.csv              # Cache des inférences directionnelles Chronos-Bolt
├── src/
│   ├── data_loader.py                 # Téléchargement et synchronisation OHLCV (CCXT / Binance)
│   ├── metrics.py                     # Calcul vectorisé : r_t, Garman-Klass, Wasserstein 1D
│   ├── primary_model.py               # DirectionalForecaster (Momentum vs Chronos-Bolt) & e_t
│   ├── circuit_breaker.py             # SurrogateDecisionTree (CART, règles if/then, signal S_t)
│   └── backtest.py                    # Moteur vectorisé net de frais & métriques de performance
├── run_all_stages.py                  # Script unifié orchestrant les Étapes 1, 2, 3 et results.png
├── run_pipeline.py                    # Pipeline de recherche et d'affichage graphique de référence
├── test_stage1.py                     # Test unitaire de validation de l'Étape 1
├── test_stage2.py                     # Test unitaire de validation de l'Étape 2
├── test_stage3.py                     # Test unitaire de validation de l'Étape 3
├── requirements.txt                   # Dépendances Python (torch, chronos, ccxt, pandas, etc.)
├── output.md                          # Journal de bord append-only des explications quantitatives
└── results.png                        # Graphique comparatif institutionnel à 3 panneaux
```

---

## 4. Fonctionnement Détaillé Composant par Composant

### 4.1. Chargement des Données : `src/data_loader.py`
- Utilise la bibliothèque `ccxt` pour interroger l'API publique de Binance sans clé secrète.
- Télécharge l'historique horaire complet (par exemple 730 jours, soit ~17 500 barres).
- Stocke les données localement dans `data/btc_1h.csv` pour assurer la reproductibilité déterministe et éviter les requêtes réseau redondantes.

### 4.2. Ingénierie des Caractéristiques : `src/metrics.py`
Ce module enrichit le DataFrame OHLCV de 3 caractéristiques vectorisées :
1. **Log-rendements continus** :
   $$r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)$$
2. **Volatilité locale de Garman-Klass (1980)** :
   $$\sigma_{GK, t}^2 = 0.5 \left[\ln\left(\frac{H_t}{L_t}\right)\right]^2 - (2\ln 2 - 1) \left[\ln\left(\frac{C_t}{O_t}\right)\right]^2$$
   Offre une mesure d'agitation instantanée sans aucun décalage temporel (*zero-lag*), en purgeant l'effet de tendance via le terme soustractif de l'Open/Close.
3. **Distance 1D de Wasserstein Glissante ($W_1$)** :
   Mesure l'écart de distribution (*Earth Mover's Distance*) entre la dynamique de fond ($W_{ref} = 200$ barres) et la dynamique immédiate ($W_{curr} = 30$ barres). Une hausse de $W_1$ signale une anomalie statistique structurelle.

### 4.3. Modèle Primaire et Résidus : `src/primary_model.py`
- **Interface Abstraite `DirectionalForecaster`** : Contrat de code garantissant qu'aucune donnée future n'est utilisée.
- **`MomentumForecaster`** : Baseline rapide basée sur la moyenne mobile des rendements passés.
- **`ChronosBoltForecaster`** : Modèle de fondation de séries temporelles pré-entraîné par Amazon. Il génère une distribution de probabilité sur 9 quantiles et compare la médiane projetée au dernier cours de clôture.
- **Calcul de la Cible Réelle et des Résidus** :
  - Cible future : $y_t = \text{sign}(r_{t+1}) \in \{-1, +1\}$
  - Résidu binaire d'erreur :
    $$e_t = \mathbb{I}(\hat{y}_t \neq y_t) = \begin{cases} 0 & \text{si le modèle primaire a vu juste} \\ 1 & \text{si le modèle primaire a échoué} \end{cases}$$

### 4.4. Le Coupe-Circuit Interprétable : `src/circuit_breaker.py`
- Entraîne un arbre de classification CART régularisé (`max_depth=2`, `min_samples_leaf=50`) pour prédire la variable cible $e_t$ à partir des caractéristiques de marché $\mathbf{X}_t = [W_{1, t}, \sigma_{GK, t}]$.
- **Extraction Symbolique** : Au lieu d'une boîte noire opaque, l'arbre extrait des règles explicites du type :
  ```text
  SI w1 > 0.0075 ET gk_vol > 0.0142 -> Probabilité d'erreur = 62.4% -> RÉGIME TOXIQUE -> S_t = 0 (Cash)
  SINON -> Probabilité d'erreur = 44.1% -> RÉGIME SAIN -> S_t = 1 (Trade autorisé)
  ```
- **Signal résultant** : $S_t \in \{0, 1\}$.

### 4.5. Le Moteur de Backtest Vectorisé : `src/backtest.py`
Exécute la simulation financière avec un réalisme institutionnel strict :
1. **Règle temporelle (*No Lookahead Bias*)** :
   $$r_{gross, t} = pos_{t-1} \cdot r_t$$
   La position décidée à la clôture de la barre précédente s'applique au rendement de la barre actuelle via `pos.shift(1)`.
2. **Déduction stricte des frictions de trading** :
   $$\Delta pos_t = |pos_t - pos_{t-1}|$$
   $$fee_t = \left(\frac{cost\_bps}{10\,000}\right) \cdot \Delta pos_t \quad (10\text{ bps} = 0.10\%)$$
3. **Rendement Net et Courbe d'Équité** :
   $$r_{net, t} = r_{gross, t} - fee_t$$
   $$\text{Equity}_t = \exp\left(\sum_{k=1}^t r_{net, k}\right)$$
4. **Métriques d'évaluation** : Ratio de Sharpe annualisé, Ratio de Calmar, Maximum Drawdown (perte maximale historique), Taux de temps passé en Cash, Win Rate.

---

## 5. Comment Exécuter et Utiliser le Projet ?

### 1. Activer l'Environnement Virtuel
Dans le terminal (PowerShell sous Windows) :
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Lancer le Pipeline Unifié
Le script `run_all_stages.py` est le point d'entrée universel :

```bash
# Exécution intégrale (Étapes 1, 2 et 3) avec le modèle Chronos-Bolt et génération de results.png
python run_all_stages.py

# Exécution rapide avec le modèle Momentum (calcul instantané sans GPU)
python run_all_stages.py --model momentum

# Tester uniquement une étape spécifique
python run_all_stages.py --stage 1
python run_all_stages.py --stage 2
python run_all_stages.py --stage 3 --cost-bps 15.0
```

### 3. Interprétation du Graphique `results.png`
Le graphique généré en haute résolution contient 3 panneaux synchronisés dans le temps :
- **Panneau 1 (Courbes d'Équité)** : Compare l'évolution du capital de la Stratégie Filtrée (en vert), de la Stratégie Brute (en orange) et du Buy & Hold (en gris). La courbe filtrée amortit les chutes brutales du marché.
- **Panneau 2 (Cours et Déclenchements du Coupe-Circuit)** : Affiche le cours de clôture du Bitcoin avec des bandes rouges ombrées représentant les périodes exactes où $S_t = 0$ (positions coupées et capital à 100% Cash).
- **Panneau 3 (Divergence de Wasserstein $W_1$)** : Montre les pics d'anomalie de distribution franchissant le seuil critique (percentile 85 en pointillé rouge), déclenchant la protection du capital.

---

## 6. Synthèse des Enseignements Financiers

1. **La préservation du capital prime sur la fréquence de trade** : Le passage en Cash lors des anomalies de Wasserstein permet de diviser significativement le Maximum Drawdown.
2. **L'impact dévastateur des frictions** : Sans coupe-circuit, une stratégie qui tente de prédire chaque heure effectue trop de rotations et voit ses gains bruts entièrement absorbés par les frais de transaction (10 à 20 bps par rotation). Le coupe-circuit réduit le turnover global.
3. **La supériorité de l'interprétabilité** : Grâce à l'arbre CART, chaque coupure de position est auditable et compréhensible par un comité des risques, contrairement aux réseaux de neurones boîte noire.
