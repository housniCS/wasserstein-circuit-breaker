# Wasserstein Circuit-Breaker: Distribution-Shift Resilience for Quantitative Trading

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Quantitative Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-emerald.svg)](#)
[![Model: Chronos--Bolt](https://img.shields.io/badge/Model-Amazon%20Chronos--Bolt-orange.svg)](https://github.com/amazon-science/chronos-forecasting)
[![Risk: OOD Circuit Breaker](https://img.shields.io/badge/Risk-100%25%20Cash%20Filter-red.svg)](#)

> **Real-time detection of Out-Of-Distribution (OOD) market regimes using rolling 1D Wasserstein distance ($W_1$) and interpretable surrogate decision trees to eliminate toxic drawdown in financial forecasting.**

---

## 1. Executive Summary & Thesis

Applying deep sequential models or time-series foundation models (e.g. Amazon Chronos-Bolt) directly to financial forecasting frequently fails due to **structural non-stationarity**:
- The statistical laws governing asset returns (volatility, skewness, fat tails, autocorrelation) mutate abruptly during market shocks and liquidity crunches.
- When distribution drift occurs, the signal-to-noise ratio collapses, causing directional models to generate **catastrophic false signals** precisely when capital preservation is critical.

Rather than fine-tuning an overfitted predictor, this project implements a **deterministic model risk management architecture**:
1. **Quantify statistical divergence in real time** between immediate market dynamics and recent baseline regimes via the **1D Wasserstein metric ($W_1$)**.
2. **Train an interpretable surrogate tree (CART)** directly on binary prediction residuals ($e_t = \mathbb{I}(\hat{y}_t \ne y_{t+1})$) conditioned on $[W_1, \sigma_{\text{GK}}]$.
3. **Trigger an automated circuit-breaker ($S_t = 0$)** that switches exposure to **100% Cash** when entering an invalidation regime, drastically curbing net-of-fees drawdowns.

---

## 2. System Architecture

```text
[ Live Binance OHLCV 1h : BTC/USDT ]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ VECTORIZED METRIC ENGINE (Pandas / SciPy)              │
│ 1. Log-returns       : r_t = ln(P_t / P_{t-1})          │
│ 2. Garman-Klass Vol  : sigma_{GK, t}                   │
│ 3. 1D Wasserstein    : W1(W_curr, W_ref) via quantiles │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ PRIMARY DIRECTIONAL MODEL (Zero-Shot / Momentum)       │
│ - Amazon Chronos-Bolt (Tiny, context=48h) or Momentum  │
│ - Output : Directional signal y_hat_t in {-1, +1}       │
│ - Binary residual : e_t = II(y_hat_t != sign(r_{t+1})) │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ INTERPRETABLE SURROGATE TREE (CART Regularized)        │
│ - DecisionTreeClassifier(max_depth=2, min_samples=50)  │
│ - Inputs : [W1, sigma_{GK}] ---> Target : e_t          │
│ - Decision rule : P(e_t = 1) >= tau => Circuit Breaker │
│ - Output : S_t in {0, 1} (0 = 100% Cash, 1 = Trade)    │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ VECTORIZED BACKTEST ENGINE (Net of Frictions)          │
│ - Transaction frictions : 10 bps (0.10%) per turnover  │
│ - Raw Strategy       : Exposure = y_hat_t              │
│ - Filtered Strategy  : Exposure = y_hat_t * S_t        │
│ - Benchmark          : Passive Buy & Hold BTC          │
└────────────────────────────────────────────────────────┘
```

---

## 3. Visual Performance & Benchmark

Running the end-to-end pipeline (`python run_all_stages.py`) generates the 3-panel institutional performance chart saved in [`results.png`](results.png):

![Benchmark and Circuit-Breaker Performance](results.png)

### Performance Breakdown:
- **Panel 1 (Top) - Cumulative Equity Curves**: Compares the benchmark Buy & Hold BTC (gray dashed), the Raw Unfiltered Strategy (orange), and the Wasserstein Filtered Strategy (emerald green).
- **Panel 2 (Middle) - Regime Cut Zones ($S_t = 0$)**: Displays the BTC price path overlaid with crimson red bars indicating exactly where the circuit breaker activated (100% Cash), successfully neutralizing flash crashes and volatile drift phases.
- **Panel 3 (Bottom) - Rolling 1D Wasserstein Divergence ($W_1$)**: Shows statistical divergence spikes that precede directional failure.

---

## 4. Mathematical Foundations

### A. Rolling 1D Wasserstein Distance ($W_1$)
To quantify distribution drift without assuming Gaussianity, we compare two sliding empirical return windows:
- **Reference Window ($W_{\text{ref}}$)**: Past $N = 200$ hours.
- **Current Window ($W_{\text{curr}}$)**: Recent $M = 30$ hours.

The optimal transport distance between empirical distributions $P_{\text{ref}}$ and $P_{\text{curr}}$ is computed via the $L_1$ norm of their empirical quantile functions $F^{-1}$:

$$W_1(P_{\text{ref}}, P_{\text{curr}}) = \int_0^1 \vert F_{\text{ref}}^{-1}(u) - F_{\text{curr}}^{-1}(u) \vert \, du$$

Computed in $\mathcal{O}(K \log K)$ using `scipy.stats.wasserstein_distance`. A sudden spike in $W_1$ indicates severe distribution distortion (tail fatness, skew shift, or volatility clustering).

### B. Garman-Klass Extreme-Value Volatility
To exploit high, low, open, and close prices rather than noisy close-to-close variations:

$$\sigma_{GK, t}^2 = \frac{1}{2} \left[ \ln\left(\frac{H_t}{L_t}\right) \right]^2 - (2\ln 2 - 1) \left[ \ln\left(\frac{C_t}{O_t}\right) \right]^2$$

### C. Binary Error Residual Target
The primary predictor produces direction $\hat{y}_t \in \{-1, +1\}$. Rather than modeling raw prices, the risk layer targets the directional misclassification indicator:

$$e_t = \mathbb{I}\left(\hat{y}_t \ne \text{sign}(r_{t+1})\right) \in \{0, 1\}$$

### D. Interpretable Surrogate Decision Tree (CART)
We fit a regularized shallow decision tree ($d=2$):

$$e_t \sim \mathcal{T}(W_{1, t}, \, \sigma_{GK, t})$$

For any leaf $l$, the empirical failure probability is:

$$p_l = P(e_t = 1 \mid x_t \in \mathcal{R}_l)$$

Given critical risk threshold $\tau$ (e.g. $52.0\%$):

$$S_t = \begin{cases} 0 \quad (\text{100\% Cash / Circuit-Breaker Triggered}) & \text{if } p_l \ge \tau \\ 1 \quad (\text{Trade Authorized}) & \text{if } p_l < \tau \end{cases}$$

### E. Realistic Frictions & Vectorized Execution
To avoid *lookahead bias*, positions decided at $t-1$ are multiplied by returns realized at $t$:

$$r_{\text{gross}, t} = \text{pos}_{t-1} \times r_t$$

Transaction costs (slippage + exchange fees, 10 bps = 0.0010) are charged upon every portfolio turnover:

$$\text{turnover}_t = |\text{pos}_{t-1} - \text{pos}_{t-2}|$$

$$r_{\text{net}, t} = r_{\text{gross}, t} - (\text{turnover}_t \times c)$$

---

## 5. Repository Structure

```text
wasserstein-circuit-breaker/
│
├── data/                               # Cached sample data for instant execution
│   ├── btc_1h.csv                      # Binance BTC/USDT 1h OHLCV
│   └── chronos_preds.csv               # Zero-shot Amazon Chronos-Bolt directional inferences
│
├── docs/                               # Comprehensive mathematical & quantitative guides
│   ├── EXPLICATION_GLOBALE_PROJET.md   # Complete project breakdown & methodology
│   ├── EXPLICATION_BACKTEST.md         # Vectorized backtest & friction mechanics
│   ├── EXPLICATION_TARGET_RESIDUALS.md # Formulation of binary failure targets
│   ├── RESIDU_ERREUR_BINAIRE.md        # Mathematical derivations of e_t
│   └── SUITE_DU_PROJET.md              # Research roadmap & prospective enhancements
│
├── src/                                # Core Python package
│   ├── __init__.py                     # Package export declarations
│   ├── data_loader.py                  # Binance REST API client + synthetic regime fallback
│   ├── metrics.py                      # Vectorized Garman-Klass & rolling Wasserstein W_1
│   ├── primary_model.py                # Chronos-Bolt foundation model & Momentum forecaster
│   ├── circuit_breaker.py              # CART surrogate tree & deterministic cash filter
│   └── backtest.py                     # Vectorized net-of-fees backtest engine
│
├── index.html                          # Interactive client-side analytics & stress-testing dashboard
├── results.png                         # 3-panel performance & drift visualization
├── run_all_stages.py                   # Master runner (Stages 1, 2, 3 + visual plot)
├── run_pipeline.py                     # Configurable CLI execution pipeline
├── test_stage1.py                      # Stage 1 unit test: Data & features
├── test_stage2.py                      # Stage 2 unit test: Surrogate CART tree
├── test_stage3.py                      # Stage 3 unit test: Vectorized backtesting
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 6. Interactive Visual Dashboard (`index.html`)

This repository includes a client-side analytics dashboard in [`index.html`](index.html).

Key features:
- **Interactive 2D Feature Space**: Real-time scatter plot of $[W_1, \sigma_{\text{GK}}]$ colored by model error $e_t$ and tree partition boundaries.
- **Tree Inspector**: Visual hierarchy of the surrogate decision tree with sample counts, Gini purity, and leaf error probabilities.
- **Interactive Stress-Testing Slider**: Dynamically adjust the critical threshold $\tau$ to see instantaneous impact on Cash allocation, equity curve, and drawdown.

Simply open `index.html` in any web browser:
```bash
# Windows
start index.html
# Linux / macOS
xdg-open index.html || open index.html
```

---

## 7. Quickstart & Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/housniCS/wasserstein-circuit-breaker.git
cd wasserstein-circuit-breaker
```

### Step 2: Create a virtual environment & install dependencies
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Run the end-to-end pipeline
```bash
python run_all_stages.py
```

### Optional CLI Arguments:
```bash
# Evaluate with Momentum baseline instead of Chronos-Bolt
python run_all_stages.py --model momentum

# Stress-test with higher frictions (e.g. 20 bps)
python run_all_stages.py --cost-bps 20.0

# Run a specific stage (1, 2, or 3)
python run_all_stages.py --stage 2
```

---

## 8. Quantitative Pitch & Talking Points

### Research & Academic Context:
> *"To address model degradation in non-stationary time series, I engineered an out-of-distribution detection protocol that quantifies distribution drift in real time via the 1D Wasserstein metric between rolling return windows. An interpretable surrogate tree trained on the predictor's misclassification residuals partitions the metric space $(W_1, \sigma_{\text{GK}})$, establishing a deterministic inference-rejection rule with near-zero compute overhead."*

### Hedge Fund & Systematic Trading Context:
> *"Machine learning alpha strategies bleed capital during regime changes when distribution drift causes false signals. By coupling rolling Wasserstein divergence and Garman-Klass volatility into a CART surrogate tree, this system detects when the market enters toxic prediction regimes and forces 100% Cash allocation. Net of 10 bps transaction costs, this circuit breaker eliminates severe tail risk while preserving upside."*

---

## 9. License

This project is licensed under the [MIT License](LICENSE).
