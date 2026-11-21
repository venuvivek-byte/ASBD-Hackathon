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
1. **Lossless Pattern Compression:** FP-Growth discovered 284 frequent itemsets; the custom CHARM algorithm pruned this down to 96 Closed Frequent Itemsets, achieving a **66.20% lossless compression ratio** ($1 - 96/284$).
2. **Cluster Validation:** AGNES hierarchical clustering isolated 3 distinct economic tiers (Recent High-Capital Takeovers, Hyper-Growth Outliers, and Historic Blue-Chips) with an exceptional **cophenetic correlation coefficient of 0.7351** (optimal $k=3$, silhouette $s=0.3764$, cut height $h=8.42$).
3. **Local Multi-Core PySpark Speedup & Crossover:** An empirical scaling benchmark across $1\times$ ($N=50$ empirical records) to $1000\times$ ($N=50,000$ synthetic replicated records) using local multi-core Spark execution (`local[*]`, thread-level parallelism on a single JVM node) demonstrated the crossover dynamics. Cold JVM initialization, Catalyst query optimization, SerDe latency, and thread context switching overhead dominate small batches ($S = 0.0024\times$ at $1\times$, noting single-node was $\sim 415\times$ faster; $S = 0.27\times$ at $10\times$), before crossing the break-even parity threshold within the $[500, 5,000]$ records interval to achieve a **$5.20\times$ parallel speedup** ($13.82\text{s}$ vs $71.85\text{s}$) and throughput of **$3,618.5\text{ records/sec}$** at $N=50,000$. Note: Workloads at $10\times$, $100\times$, and $1,000\times$ are synthetic replicated scaling experiments to evaluate multi-core computational throughput, not additional independent real-world sports teams.
4. **Academic Publication:** Formal 8-page academic publication PDF (`report/ASBD_Hackathon_Final_Report.pdf`) and comprehensive 3.15 MB editable academic Word document (`report/ASBD_Sports_Analytics_Academic_Report.docx`) featuring 25 structured tables, embedded 300 DPI figures, and rigorous statistical inferences.

---

## 📂 Repository Structure

```
ASBD-Hackathon/
├── README.md                                # Master documentation and run guide
├── data/
│   ├── raw_sports_teams.csv                 # Raw Forbes Top 50 sports teams dataset
│   ├── clean.csv                            # Cleaned, imputed, and feature-engineered dataset
│   ├── scaled.csv                           # Standardized (Z-score & MinMax) features
│   ├── transactions.csv                     # Discretized transaction token itemsets
│   └── archive.zip                          # Original Kaggle archive
├── notebooks/
│   ├── 01_load_and_inspect.ipynb            # Phase 1: Schema audit and missing value diagnostics
│   ├── 02_preprocessing.ipynb               # Phase 2: Imputation, IQR outliers, feature engineering
│   ├── 03_descriptive_analytics.ipynb       # Phase 3: Exploratory visualizations & statistical distributions
│   ├── 04_predictive_and_mining.ipynb       # Phase 4: FP-Growth, CHARM CFI, Naive Bayes & AGNES
│   ├── 05_pyspark_benchmark.ipynb           # Phase 5: Distributed PySpark vs Single-Node scaling benchmark
│   ├── 06_final_report.ipynb                # Phase 6: Master executive summary report notebook
│   ├── ASBD_PySpark_Colab_Benchmark.ipynb   # 1-Click self-contained Google Colab benchmark notebook
│   └── sports_analytics_pipeline.ipynb      # End-to-end monolithic pipeline notebook
├── outputs/
│   ├── ASBD_Hackathon_Submission.zip        # Standalone submission archive (7.9 MB, 58 items)
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
│       ├── fp_growth_rules.csv              # 836 Association Rules (318 minimal non-redundant rules)
│       ├── charm_cfis.csv                   # 96 Closed Frequent Itemsets from CHARM
│       ├── charm_comparison.json            # 66.20% CFI Compression Ratio evaluation
│       ├── naive_bayes_results.json         # Naive Bayes metrics, confusion matrix, decision rules
│       ├── agnes_summary.json               # Cophenetic score, silhouette score, cluster cut height
│       ├── agnes_cluster_profiles.csv       # Multi-dimensional cluster profile statistics
│       ├── pyspark_benchmark_results.json   # Benchmark timing, hardware specs, speedup metrics
│       ├── pyspark_benchmark_table.csv      # Scaled runtime table across 1x, 10x, 100x, 1000x
│       └── *.json                           # Structured audit validation manifests
├── report/
│   ├── ASBD_Hackathon_Final_Report.md       # Comprehensive publication-grade Markdown report
│   ├── ASBD_Hackathon_Final_Report.pdf      # Formal 8-page academic publication PDF
│   ├── ASBD_Sports_Analytics_Academic_Report.docx # The One Final Editable Word Report (3.15 MB)
│   └── INDEPENDENT_VALIDATION_REPORT.md     # Comprehensive auditor validation report
└── src/
    ├── phase1_inspect.py                    # Phase 1: Data ingestion & schema inspection
    ├── phase2_preprocess.py                 # Phase 2: Cleaning, imputation, IQR flags & scaling
    ├── phase3_visualizations.py             # Phase 3: Generates all 15 publication-grade figures
    ├── phase4_mining.py                     # Phase 4: Leak-free Naive Bayes & AGNES clustering
    ├── phase5_pyspark_benchmark.py          # Phase 5: PySpark distributed benchmark runner
    ├── charm.py                             # Custom CHARM algorithm (IT-Tree CFI miner)
    ├── fpgrowth.py                          # Custom FP-Growth frequent pattern mining engine
    ├── generate_academic_docx.py            # Word document (.docx) generation module
    ├── generate_academic_pdf.py             # PDF publication generation module
    └── package_submission.py                # Submission archive bundler
```

---

## ⚙️ System Requirements & Installation

### Core Prerequisites
- **Python:** 3.10 or higher
- **Java:** OpenJDK 11 or 17 LTS (required for PySpark execution)
- **OS:** Windows 10/11, macOS, or Linux

### Python Dependencies
Install required packages using pip:
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn reportlab pillow psutil nbformat mlxtend
```

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

# Step 6: Compile the academic Word document and PDF report
python src/generate_academic_docx.py
python src/generate_academic_pdf.py

# Step 7: Bundle the final submission archive
python src/package_submission.py
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
5. `05_pyspark_benchmark.ipynb`: Scaling benchmark, multi-core worker partitioning, and speedup crossover analysis.
6. `06_final_report.ipynb`: Executive portfolio notebook integrating all metrics, tables, and strategic recommendations.
7. `ASBD_PySpark_Colab_Benchmark.ipynb`: Self-contained 1-click Google Colab benchmark notebook.
8. `sports_analytics_pipeline.ipynb`: Monolithic end-to-end interactive notebook.

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
- **Plot 02:** Valuation spread by sport (European Soccer exhibits highest median at $\$4.10\text{B}$; NFL exhibits highest market share at $52\%$ and uniform floor).
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
- **Plot 13:** Naive Bayes valuation tier confusion matrix across High, Med, and Low tiers (50.0% test accuracy, 0.5000 Macro F1).
- **Plot 14:** AGNES hierarchical dendrogram (Ward linkage, cophenetic $r=0.7351$, cut-height threshold $h = 8.42$, $k=3$).
- **Plot 15:** Distributed speedup and execution runtime scaling diagnostic across $1\times$ to $1000\times$ scales, showing crossover interval in $[500, 5,000]$ records.

### Phase 4: Pattern Mining & Predictive Modeling
- **FP-Growth:** Discovered 284 frequent itemsets and 836 association rules ($\text{min\_support}=0.10, \text{min\_confidence}=0.60$), with 318 minimal rules after redundancy pruning (61.96% compression).
- **CHARM CFI Algorithm:** Extracted 96 Closed Frequent Itemsets, achieving a **$66.20\%$ lossless compression ratio** ($1 - 96/284$).
- **Naive Bayes Classification:** Evaluated on stratified hold-out test partition with Laplace smoothing ($\alpha=1.0$); achieved $50.0\%$ test accuracy and $0.5000$ Macro F1 with errors restricted strictly to adjacent tiers. Decision rules extracted via class conditional probabilities.
- **AGNES Hierarchical Clustering:** Ward's linkage achieved a **$0.7351$ cophenetic correlation coefficient**, cleanly partitioning the franchises into 3 macro-clusters at cut height $h=8.42$ ($s=0.3764$):
  1. *Recent High-Capital Takeovers ($N=7$):* Mean price paid $\$2,214\text{M}$, mean tenure $6.00\text{ yrs}$.
  2. *Hyper-Growth Outliers ($N=4$):* Mean 5-yr growth $173.8\%$, mean tenure $10.75\text{ yrs}$.
  3. *Historic Blue-Chip Core ($N=39$):* Mean value $\$3.47\text{B}$, mean tenure $33.72\text{ yrs}$.

### Phase 5: Distributed Benchmark & Speedup Analysis
Evaluated in Google Colab using `local[*]` mode—**local multi-core Spark execution using thread-level parallelism on a single JVM node**, NOT a multi-machine distributed cluster. The benchmark evaluated native `pyspark.ml.fpm.FPGrowth` and `pyspark.ml.classification.NaiveBayes` against single-node Scikit-Learn/mlxtend:

| Scale Factor | Records ($N$) | Single-Node (s) | PySpark `local[*]` (s) | Speedup Ratio ($S$) | Distributed Throughput | Computing Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1x** | 50 | 0.0612 s | 25.4210 s | **0.0024x** | 2.0 rec/s | Overhead Dominated (Cold JVM, ~415x Single-Node) |
| **10x** | 500 | 1.4820 s | 5.4120 s | **0.27x** | 92.4 rec/s | Overhead Dominated |
| **100x** | 5,000 | 6.8450 s | 4.1980 s | **1.63x** | 1,191.0 rec/s | Parallel Accelerated (Crossover) |
| **1000x** | 50,000 | 71.8540 s | 13.8180 s | **5.20x** | **3,618.5 rec/s** | **Massive Parallel Speedup (5.20x)** |

**Key Takeaways & Boundary Constraints:**
1. **Local Multi-Core Architecture:** PySpark execution was conducted in `local[*]` mode utilizing thread-level parallelism on a single JVM node, not across a multi-node physical distributed cluster.
2. **Small-Data Overhead Penalty ($N \le 500$):** At small scales, single-node execution significantly outperforms Spark ($0.0024\times$ at $1\times$, noting single-node was $\sim 415\times$ faster; $0.27\times$ at $10\times$) due to fixed initialization, cold JVM startup, Catalyst optimizer query planning, SerDe latency, and thread context switching overhead (~5.4s to 25.4s).
3. **Synthetic Scaling Limitations:** The original Forbes dataset comprises 50 empirical records. The $10\times$ (500), $100\times$ (5,000), and $1,000\times$ (50,000) workloads are synthetic replicated scaling experiments to evaluate multi-core computational throughput, not additional independent real-world sports teams.
4. **Empirical Crossover Interval:** Computational complexity scales super-linearly with $N$ while coordination overhead remains constant. The crossover point is observed strictly within the **$[500, 5,000]$** records interval. At $100\times$ ($N=5,000$), PySpark achieves a $1.63\times$ speedup, and at $1,000\times$ ($N=50,000$), it achieves a $5.20\times$ speedup ($13.82\text{s}$ vs $71.85\text{s}$).

### Phase 6: Final Report & Submission Package
- **Academic Publication Markdown:** `report/ASBD_Hackathon_Final_Report.md`
- **Academic Publication PDF:** `report/ASBD_Hackathon_Final_Report.pdf` (8 pages, formal layout, embedded figures and tables)
- **Academic Word Report:** `report/ASBD_Sports_Analytics_Academic_Report.docx` (3.15 MB, editable document with 25 tables and 15 figures)
- **Independent Validation Report:** `report/INDEPENDENT_VALIDATION_REPORT.md` (Full audit trail and rubric score)
- **Executive Portfolio Notebook:** `notebooks/06_final_report.ipynb`
- **Submission Archive:** `outputs/ASBD_Hackathon_Submission.zip` (7.9 MB, 58 files)

---

## 💻 Hardware & Software Specifications
- **Operating System:** Windows 11 Enterprise (Build 26300)
- **Processor:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
- **Logical CPU Cores:** 16 Cores (12 Physical Cores)
- **Host Memory:** 15.69 GB RAM
- **Runtime Environments:** Python 3.10.0, Java 17.0.15 LTS
- **Core Libraries:** `numpy 1.26.4`, `pandas 2.2.2`, `scikit-learn 1.7.2`, `scipy 1.15.3`, `seaborn 0.13.2`, `matplotlib 3.8.4`, `reportlab 5.1.0`, `pillow 11.2.1`, `mlxtend 0.23.4`

---

## 🎯 Verification & Reproducibility Checklist
To guarantee 100% end-to-end reproducibility:
- [x] Deterministic seed (`random_state = 42`) set across all scripts and notebooks.
- [x] Zero data leakage: Discretization thresholds and encoders fitted exclusively on train split.
- [x] Lossless pattern compression verified ($|\text{CFI}| = 96$ vs $|\text{FI}| = 284$, ratio $= 66.20\%$).
- [x] Cophenetic correlation verified ($c = 0.7351$).
- [x] All 15 figures verified in `outputs/plots/` and indexed in `inferences.json`.
- [x] Master PDF verified in `report/ASBD_Hackathon_Final_Report.pdf` (8 pages, 300 DPI).
- [x] Master Word report verified in `report/ASBD_Sports_Analytics_Academic_Report.docx` (3.15 MB).
- [x] All 8 notebooks verified in `notebooks/`.
- [x] Full submission package packaged in `outputs/ASBD_Hackathon_Submission.zip` (7.9 MB, 58 items).

---

*Authored by the ASBD Hackathon Research Team — November 2026.*
