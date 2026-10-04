# Explication Mathématique et Pratique : `compute_directional_target_and_residuals`

Ce document détaille le fonctionnement, les fondements mathématiques en LaTeX et l'utilité quantitative de la fonction `compute_directional_target_and_residuals` présente dans [`src/primary_model.py`](src/primary_model.py).

---

## 1. Vue d'ensemble et Objectif

La fonction a pour rôle d'évaluer le **modèle primaire** et de construire le **résidu d'erreur binaire** à chaque pas de temps $t$.

Dans une stratégie quantitative à deux étages (architecture inspirée du *méta-labelling* de Marcos López de Prado) :
- Le **modèle primaire** tente de prédire la direction du marché.
- Le **méta-modèle** (modèle secondaire) tente d'anticiper les moments où le modèle primaire va commettre une erreur pour calibrer la taille de position ou filtrer les faux signaux.

---

## 2. Décomposition pas à pas avec Formules LaTeX

### Étape 1 : Inférence directionnelle à l'instant $t$

```python
df_out["pred_dir"] = forecaster.predict_direction(df_out)
```

#### Formulation Mathématique
À chaque instant $t$, le modèle primaire reçoit le vecteur de caractéristiques de marché disponibles $X_t$ et produit une prédiction directionnelle notée $\hat{y}_t$ :

$$\hat{y}_t \in \{-1, +1\}$$

Où :
- $\hat{y}_t = +1$ correspond à un signal d'achat (*long* / anticipation d'une hausse).
- $\hat{y}_t = -1$ correspond à un signal de vente (*short* / anticipation d'une baisse).

---

### Étape 2 : Cible réelle future à $t+1$ et décalage temporel

```python
next_return = df_out["ret"].shift(-1)
target_dir = np.where(next_return >= 0, 1, -1)
df_out["target_dir"] = np.where(next_return.isna(), np.nan, target_dir)
```

#### Formulation Mathématique
Soit $r_{t+1}$ le rendement arithmétique (ou logarithmique) réalisé sur la période future $[t, t+1]$ :

$$r_{t+1} = \frac{P_{t+1} - P_t}{P_t}$$

La direction réelle future du marché $y_t$ associée à la décision prise à $t$ est définie par la fonction signe du rendement futur :

$$y_t = \text{sign}(r_{t+1}) = \begin{cases} +1 & \text{si } r_{t+1} \ge 0 \\ -1 & \text{si } r_{t+1} < 0 \end{cases}$$

#### Pourquoi utiliser `.shift(-1)` ?
- Dans le DataFrame, la ligne courante $t$ contient les informations connues au temps $t$.
- Le rendement effectif qui découle de la décision prise en $t$ se réalise à la période suivante ($t+1$).
- L'opération `.shift(-1)` remonte la valeur de la ligne suivante ($t+1$) sur la ligne actuelle ($t$) pour aligner la prédiction passée avec le résultat futur correspondant.

#### Condition aux limites (dernière ligne)
Pour la dernière ligne de la série temporelle :
$$r_{T+1} = \text{inconnu} \implies y_T = \text{NaN}$$

---

### Étape 3 : Résidu binaire d'erreur ($e_t$)

```python
error = (df_out["pred_dir"] != df_out["target_dir"]).astype(float)
df_out["model_error"] = np.where(df_out["target_dir"].isna(), np.nan, error)
```

#### Formulation Mathématique
On définit le résidu d'erreur $e_t$ via la fonction indicatrice $\mathbb{I}(\cdot)$ :

$$e_t = \mathbb{I}(\hat{y}_t \neq y_t) = \begin{cases} 1 & \text{si } \hat{y}_t \neq y_t \quad (\text{erreur de prédiction}) \\ 0 & \text{si } \hat{y}_t = y_t \quad (\text{prédiction correcte}) \end{cases}$$

Ou sous forme algébrique équivalente :

$$e_t = \frac{1 - \hat{y}_t \cdot y_t}{2}$$

En effet :
- Si $\hat{y}_t = y_t \implies \hat{y}_t \cdot y_t = 1 \implies e_t = \frac{1 - 1}{2} = 0$
- Si $\hat{y}_t \neq y_t \implies \hat{y}_t \cdot y_t = -1 \implies e_t = \frac{1 - (-1)}{2} = 1$

#### Traitement des valeurs manquantes
Si $y_t = \text{NaN}$ (dernière ligne), alors :
$$e_t = \text{NaN}$$

---

## 3. Tableau Illustratif

| Date ($t$) | Rendement $r_t$ | Rendement futur $r_{t+1}$ (`shift(-1)`) | Prédiction $\hat{y}_t$ (`pred_dir`) | Cible réelle $y_t$ (`target_dir`) | Erreur $e_t$ (`model_error`) | Interprétation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$t_1$** | $+0.4\%$ | $+1.5\%$ | $+1$ | $+1$ | **$0$** | Prédiction correcte (hausse anticipée et réalisée) |
| **$t_2$** | $+1.5\%$ | $-0.9\%$ | $+1$ | $-1$ | **$1$** | **Erreur** : signal d'achat alors que le marché baisse |
| **$t_3$** | $-0.9\%$ | $-0.4\%$ | $-1$ | $-1$ | **$0$** | Prédiction correcte (baisse anticipée et réalisée) |
| **$t_4$** | $-0.4\%$ | $+0.8\%$ | $-1$ | $+1$ | **$1$** | **Erreur** : signal de vente alors que le marché monte |
| **$t_5$** | $+0.8\%$ | Inconnu (`NaN`) | $+1$ | `NaN` | `NaN` | Échéance future non encore observée |

---

## 4. Rôle dans l'Architecture Quantitative

La variable générée `model_error` ($e_t$) constitue la **variable cible** ($Y$) pour l'entraînement du méta-modèle :

$$\mathcal{M}_{\text{meta}} : X_t \longmapsto \mathbb{P}(e_t = 1 \mid X_t)$$

- Si $\mathbb{P}(e_t = 1 \mid X_t) \ge \tau$ (seuil de risque d'erreur élevé) :
  $$\text{Action} = 0 \quad (\text{abstention / taille de position nulle})$$
- Si $\mathbb{P}(e_t = 1 \mid X_t) < \tau$ (confiance élevée dans le modèle primaire) :
  $$\text{Action} = \hat{y}_t \quad (\text{exécution du signal})$$
