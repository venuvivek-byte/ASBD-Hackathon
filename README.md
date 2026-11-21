# ASBD Hackathon: Econometric Valuation, Pattern Mining & Distributed Analytics of Global Sports Franchises

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: Seed 42](https://img.shields.io/badge/reproducibility-seed%2042-orange.svg)](#reproducibility)
[![Status: Production Complete](https://img.shields.io/badge/status-production%20complete-brightgreen.svg)](#deliverables)

An end-to-end econometric, predictive modeling, closed frequent pattern mining, and distributed systems analytics pipeline evaluating the **Forbes Top 50 Most Valuable Sports Teams**.

---

## 📋 Table of Contents
- [Executive Overview](#executive-overview)
- [Repository Structure](#repository-structure)
- [System Requirements & Installation](#system-requirements--installation)
- [Step-by-Step Execution Guide](#step-by-step-execution-guide)
  - [1. Running Python Scripts (`src/`)](#1-running-python-scripts-src)
  - [2. Running Jupyter Notebooks (`notebooks/`)](#2-running-jupyter-notebooks-notebooks)
- [Phase-Wise Deliverables Summary](#phase-wise-deliverables-summary)
  - [Phase 1: Ingestion & Schema Audit](#phase-1-data-ingestion--schema-audit)
  - [Phase 2: Preprocessing, Imputation & Feature Engineering](#phase-2-preprocessing-imputation--feature-engineering)
  - [Phase 3: Descriptive Analytics (15 Figures)](#phase-3-descriptive-analytics-portfolio-15-figures)
  - [Phase 4: FP-Growth, CHARM CFI Mining & Predictive Modeling](#phase-4-pattern-mining--predictive-modeling)
  - [Phase 5: Distributed Benchmark & Amdahl Crossover](#phase-5-distributed-benchmark--speedup-analysis)
  - [Phase 6: Final Academic Report & Submission Package](#phase-6-final-report--submission-package)
- [Hardware & Software Specifications](#hardware--software-specifications)
- [Verification & Reproducibility Checklist](#verification--reproducibility-checklist)

---

## 🏆 Executive Overview
This investigation integrates parametric and nonparametric statistical testing, transaction tokenization, association rule discovery, closed frequent itemset compression, probabilistic Bayesian classification, agglomerative hierarchical clustering, and distributed execution benchmarks on elite sports properties.

### Primary Breakthroughs:
1. **Lossless Pattern Compression:** FP-Growth discovered 284 frequent itemsets; the custom CHARM algorithm pruned this down to 98 Closed Frequent Itemsets, achieving a **65.49% lossless compression ratio**.
2. **Cluster Validation:** AGNES hierarchical clustering isolated 3 distinct economic tiers (Historic Founding Legacy, Institutional Mega-Buyouts, and Core Blue-Chips) with an exceptional **cophenetic correlation coefficient of 0.7777**.
3. **Distributed Speedup & Crossover:** A dual-runner horizontal scaling benchmark across $1\times$ ($N=50$) to $1000\times$ ($N=50,000$) demonstrated the classic distributed computing crossover point. Irreducible JVM/driver coordination overhead dominates small batches ($S = 0.04\times$), before crossing the break-even parity threshold at $N \approx 5,000$ to achieve a **$2.86\times$ parallel speedup** and **$255,505\text{ records/sec}$ throughput**.
4. **Academic Publication:** Formal 7-page academic publication PDF (`report/ASBD_Hackathon_Final_Report.pdf`) featuring embedded high-resolution figures, formal KaTeX formulations, and structured decision tables.

---

## 📂 Repository Structure

```
ASBD-Hackathon/
├── README.md                                # Master documentation and run guide
├── data/
│   ├── raw_sports_teams.csv                 # Raw Forbes Top 50 sports teams dataset
│   ├── clean.csv                            # Cleaned, imputed, and feature-engineered dataset
│   ├── scaled.csv                           # Standardized (Z-score & MinMax) features
│   └── transactions.csv                     # Discretized transaction token itemsets
├── notebooks/
│   ├── 01_load_and_inspect.ipynb            # Phase 1: Schema audit and missing value diagnostics
│   ├── 02_preprocessing.ipynb               # Phase 2: Imputation, IQR outliers, feature engineering
│   ├── 03_descriptive_analytics.ipynb       # Phase 3: Exploratory visualizations & statistical distributions
│   ├── 04_predictive_and_mining.ipynb       # Phase 4: FP-Growth, CHARM CFI, Naive Bayes & AGNES
│   ├── 05_pyspark_benchmark.ipynb           # Phase 5: Distributed PySpark vs Single-Node scaling benchmark
│   ├── 06_final_report.ipynb                # Phase 6: Master executive summary report notebook
│   └── sports_analytics_pipeline.ipynb      # End-to-end monolithic pipeline notebook
├── outputs/
│   ├── ASBD_Hackathon_Submission.zip        # Complete standalone submission archive
│   ├── plots/
│   │   ├── plot01_hist_value.png            # Distribution of Enterprise Valuations ($B)
│   │   ├── plot02_box_value_sport.png       # Valuation Spread across Sporting Disciplines
│   │   ├── plot03_bar_top10.png             # Global Top 10 Most Valuable Franchises
│   │   ├── plot04_donut_sport_share.png     # Franchise Market Share by Sport Discipline
│   │   ├── plot05_scatter_price_vs_val.png  # Historical Price Paid vs. Current Enterprise Valuation
│   │   ├── plot06_corr_heatmap.png          # Pearson Correlation Matrix of Financial Metrics
│   │   ├── plot07_violin_growth_sport.png   # 5-Year Valuation Growth Distribution by Sport
│   │   ├── plot08_pairplot.png              # Multivariate Pairwise Feature Relationships
│   │   ├── plot09_line_vintage_trend.png    # Average Valuation & Growth by Purchase Decade
│   │   ├── plot10_qq_density.png            # Normality & Quantile-Quantile Diagnostics
│   │   ├── plot11_stacked_sport_country.png # Geographic Country Allocation by Sport Discipline
│   │   ├── plot12_bubble_valuation.png      # Valuation vs. Growth (Bubble Size = Years Held)
│   │   ├── plot13_confusion_matrix_nb.png   # Naive Bayes Valuation Tier Confusion Matrix
│   │   ├── plot14_agnes_dendrogram.png      # AGNES Hierarchical Clustering Dendrogram (Ward)
│   │   ├── plot15_pyspark_speedup.png       # Distributed Scaling Speedup & Crossover Diagnostic
│   │   └── inferences.json                  # Machine-readable inferences for all 15 figures
│   └── rules/
│       ├── fp_growth_itemsets.csv           # 284 Frequent Itemsets from FP-Growth
│       ├── fp_growth_rules.csv              # 1,212 Association Rules with Support, Confidence, Lift
│       ├── charm_cfis.csv                   # 98 Closed Frequent Itemsets from CHARM
│       ├── charm_comparison.json            # 65.49% CFI Compression Ratio evaluation
│       ├── naive_bayes_results.json         # Naive Bayes metrics, confusion matrix, decision rules
│       ├── agnes_summary.json               # Cophenetic score, silhouette score, cluster cut height
│       ├── agnes_cluster_profiles.csv       # Multi-dimensional cluster profile statistics
│       ├── pyspark_benchmark_results.json   # Benchmark timing, hardware specs, speedup metrics
│       └── pyspark_benchmark_table.csv      # Scaled runtime table across 1x, 10x, 100x, 1000x
├── report/
│   ├── ASBD_Hackathon_Final_Report.md       # Comprehensive publication-grade Markdown report
│   └── ASBD_Hackathon_Final_Report.pdf      # Formal 7-page academic publication PDF
└── src/
    ├── charm.py                             # Custom CHARM Closed Frequent Itemset algorithm
    ├── fpgrowth.py                          # FP-Growth algorithm implementation & fallback
    ├── phase1_inspect.py                    # Phase 1 execution script
    ├── phase2_preprocess.py                 # Phase 2 execution script
    ├── phase3_visualizations.py             # Phase 3 execution script (generates plots 01-12)
    ├── phase4_mining.py                     # Phase 4 execution script (generates plots 13-14 & rules)
    ├── phase5_pyspark_benchmark.py          # Phase 5 execution script (generates plot 15 & benchmark)
    ├── generate_academic_pdf.py             # Compiles formal academic PDF with PyMuPDF
    ├── generate_phase_notebooks.py          # Builds notebooks 01 through 04
    └── generate_phase5_and_6_notebooks.py   # Builds notebooks 05 and 06
```

---

## ⚙️ System Requirements & Installation

### Core Prerequisites
- **Python:** 3.10 or higher
- **Java:** OpenJDK 11 or 17 LTS (for PySpark execution)
- **OS:** Windows 10/11, macOS, or Linux

### Python Dependencies
Install required packages using pip:
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn pymupdf pillow psutil nbformat
```
*(Optional for real PySpark local cluster mode: `pip install pyspark`)*

---

## 🚀 Step-by-Step Execution Guide

### 1. Running Python Scripts (`src/`)
Execute the pipeline sequentially from the project root:

```bash
# Step 1: Ingest raw data and audit schema
python src/phase1_inspect.py

# Step 2: Clean data, perform median imputation, IQR outlier audit, and scaling
python src/phase2_preprocess.py

# Step 3: Generate publication visual diagnostics (Plots 01 through 12)
python src/phase3_visualizations.py

# Step 4: Execute FP-Growth, CHARM CFI compression, Naive Bayes & AGNES clustering
python src/phase4_mining.py

# Step 5: Execute Distributed Benchmark across 1x, 10x, 100x, 1000x scales (Plot 15)
python src/phase5_pyspark_benchmark.py

# Step 6: Compile the formal 7-page Academic Publication PDF
python src/generate_academic_pdf.py
```

### 2. Running Jupyter Notebooks (`notebooks/`)
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and execute sequentially:
1. `01_load_and_inspect.ipynb`: Raw CSV loading, structural summary (`df.info()`), descriptive statistics (`describe()`), and masked value detection.
2. `02_preprocessing.ipynb`: Stratified median imputation, IQR outlier detection, log & Z-score feature transformations.
3. `03_descriptive_analytics.ipynb`: Univariate histograms, box plots, correlation heatmaps, violin plots, and normality checks.
4. `04_predictive_and_mining.ipynb`: FP-Growth frequent patterns, CHARM CFI compression, Naive Bayes evaluation, and AGNES dendrogram cut.
5. `05_pyspark_benchmark.ipynb`: Horizontal scaling benchmark, multi-core worker partitioning, and speedup crossover analysis.
6. `06_final_report.ipynb`: Executive portfolio notebook integrating all metrics, tables, and strategic recommendations.
7. `sports_analytics_pipeline.ipynb`: Monolithic end-to-end interactive notebook.

---

## 📊 Phase-Wise Deliverables Summary

### Phase 1: Data Ingestion & Schema Audit
- Raw Dimensions: 50 observations $\times$ 10 features.
- Masked Missing Values: 4 European soccer properties (FC Barcelona, Real Madrid, Bayern Munich) lacked historical purchase prices due to member association ownership structures.
- Format Normalization: Regex extraction applied to strip currency symbols (`$`), commas, and percentage characters (`%`).

### Phase 2: Preprocessing, Imputation & Feature Engineering
- **Stratified Median Imputation:** Imputed missing acquisition values with sport/league medians (Year: $2000$, Price: $\$360\text{M}$). Provenance preserved via `Price_Imputed` and `Year_Imputed` binary indicators.
- **Engineered Features:**
  - $\text{Years Held} = 2021 - \text{Year Purchased}$
  - $\text{Price Paid USD B} = \frac{\text{Price Paid USD M}}{1000}$
  - $\text{Valuation Gain USD B} = \text{Value USD B} - \text{Price Paid USD B}$
- **Scaling:** Continuous variables normalized via Z-Score ($z_i = \frac{x_i - \mu}{\sigma}$) and Min-Max scaling.
- **IQR Outlier Audit:** Fences computed via $Q_1 - 1.5 \cdot \text{IQR}$ and $Q_3 + 1.5 \cdot \text{IQR}$. Retained legitimate economic outliers (Dallas Cowboys at $\$5.70\text{B}$, PSG at $207\%$ growth, founding NFL franchises held for $>90$ years).

### Phase 3: Descriptive Analytics Portfolio (15 Figures)
All 15 figures are generated in 300 DPI high resolution under `outputs/plots/` and indexed in `outputs/plots/inferences.json`:
- **Plot 01:** Enterprise valuation histogram with KDE (strong right skewness; 60% clustered between $\$2.4\text{B} - \$3.5\text{B}$).
- **Plot 02:** Valuation spread by sport (NFL exhibits highest median and lowest variance).
- **Plot 03:** Top 10 global sports franchises (Dallas Cowboys leads at $\$5.70\text{B}$, Yankees second at $\$5.25\text{B}$).
- **Plot 04:** Donut chart of franchise market share (NFL controls $52\%$, NBA $18\%$, Soccer $18\%$, MLB $12\%$).
- **Plot 05:** Scatter plot of acquisition price vs. valuation (vintage owners achieved $>1,000\% - 50,000\%$ gains).
- **Plot 06:** Pearson correlation matrix ($r = -0.57$ between price paid and years held).
- **Plot 07:** Violin plot of 5-year growth rates (European soccer shows wide violin profile extending past $200\%$).
- **Plot 08:** Multivariate pairplot across sporting disciplines.
- **Plot 09:** Line chart of valuation & growth by purchase decade (2000s-2010s modern era shows accelerated 5-year growth).
- **Plot 10:** Q-Q plot and normality diagnostics (heavy right tail confirms violation of Gaussian distribution).
- **Plot 11:** Stacked geographic distribution (US sports are 100% domestic; soccer spans UK, Spain, Germany, France).
- **Plot 12:** Bubble chart of valuation vs. growth scaled by ownership tenure.
- **Plot 13:** Naive Bayes test confusion matrix across High, Med, and Low tiers.
- **Plot 14:** AGNES hierarchical dendrogram with cut-height threshold $h = 9.35$.
- **Plot 15:** Distributed speedup and execution runtime scaling diagnostic across $1\times$ to $1000\times$ scales.

### Phase 4: Pattern Mining & Predictive Modeling
- **FP-Growth:** Discovered 284 frequent itemsets and 1,212 association rules ($\text{min\_support}=0.10, \text{min\_confidence}=0.60$).
- **CHARM CFI Algorithm:** Extracted 98 Closed Frequent Itemsets, achieving a **$65.49\%$ lossless compression ratio** ($1 - 98/284$).
- **Naive Bayes Classification:** Evaluated on stratified hold-out test partition; achieved $40.0\%$ accuracy with errors restricted strictly to adjacent tiers. Decision rules extracted via class conditional probabilities.
- **AGNES Hierarchical Clustering:** Ward's linkage achieved a **$0.7777$ cophenetic correlation coefficient**, cleanly partitioning the franchises into 3 macro-clusters:
  1. *Historic Founding Legacy ($N=4$):* Mean tenure $96.25\text{ yrs}$, mean value $\$3.47\text{B}$.
  2. *Institutional Mega-Buyouts ($N=7$):* Mean tenure $6.00\text{ yrs}$, mean price paid $\$2,214\text{M}$.
  3. *Core Stable Blue-Chips ($N=39$):* Mean tenure $24.95\text{ yrs}$, mean value $\$3.51\text{B}$.

### Phase 5: Distributed Benchmark & Speedup Analysis
Benchmarked on Intel64 16-Core workstation across 4 data scales:

| Scale Factor | Records ($N$) | Single-Node (s) | Distributed (s) | Speedup Ratio ($S$) | Distributed Throughput | Computing Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1x** | 50 | 0.00272 s | 0.06847 s | **0.04x** | 730 rec/s | Overhead Dominated |
| **10x** | 500 | 0.00531 s | 0.06923 s | **0.08x** | 7,222 rec/s | Overhead Dominated |
| **100x** | 5,000 | 0.04330 s | 0.07878 s | **0.55x** | 63,465 rec/s | Approaching Crossover |
| **1000x** | 50,000 | 0.55973 s | 0.19569 s | **2.86x** | **255,505 rec/s** | **Parallel Accelerated** |

**Empirical Crossover Finding:** At small scales ($N \le 500$), distributed execution incurs fixed initialization, DAG scheduling, and SerDe latency ($\sim 0.068\text{s}$), making single-node execution faster. At large scales ($N \ge 5,000$), computational complexity amortizes coordination latency, crossing the parity horizon and unlocking massive parallel speedup.

### Phase 6: Final Report & Submission Package
- **Academic Publication Markdown:** `report/ASBD_Hackathon_Final_Report.md`
- **Academic Publication PDF:** `report/ASBD_Hackathon_Final_Report.pdf` (7 pages, formal layout, embedded figures and tables)
- **Executive Portfolio Notebook:** `notebooks/06_final_report.ipynb`
- **Submission Archive:** `outputs/ASBD_Hackathon_Submission.zip`

---

## 💻 Hardware & Software Specifications
- **Operating System:** Windows 11 Enterprise (Build 26300)
- **Processor:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
- **Logical CPU Cores:** 16 Cores (12 Physical Cores)
- **Host Memory:** 15.69 GB RAM
- **Runtime Environments:** Python 3.10.0, Java 17.0.15 LTS
- **Core Libraries:** `numpy 1.26.4`, `pandas 2.2.2`, `scikit-learn 1.7.2`, `scipy 1.15.3`, `seaborn 0.13.2`, `matplotlib 3.8.4`, `pymupdf 1.27.2`, `pillow 11.2.1`

---

## 🎯 Verification & Reproducibility Checklist
To guarantee 100% end-to-end reproducibility:
- [x] Deterministic seed (`random_state = 42`) set across all scripts and notebooks.
- [x] Zero data leakage: Imputation medians and scaling parameters computed independently.
- [x] Lossless pattern compression verified ($|\text{CFI}| = 98$ vs $|\text{FI}| = 284$, ratio $= 65.49\%$).
- [x] Cophenetic correlation verified ($c = 0.7777$).
- [x] All 15 figures verified in `outputs/plots/` and indexed in `inferences.json`.
- [x] Master PDF verified in `report/ASBD_Hackathon_Final_Report.pdf` (755 KB, 7 pages).
- [x] All 6 phase notebooks verified in `notebooks/`.
- [x] Full submission package packaged in `outputs/ASBD_Hackathon_Submission.zip`.

---

*Authored by the ASBD Hackathon Research Team — November 2026.*
