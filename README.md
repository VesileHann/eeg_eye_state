# EEG Eye State Dataset Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CSE555 Final Project Report & Code Implementation**  
> **Author:** Vesile Han  
> **Dataset:** [UCI Machine Learning Repository: EEG Eye State (ID: 264)](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)

---

## 📌 Project Overview

This repository contains an in-depth statistical, dimensionality reduction, clustering, and hypothesis testing analysis of the **EEG Eye State Dataset**. 

Electroencephalography (EEG) records continuous electrical brain activity via scalp electrodes. The central objective of this study is to predict eye state (binary classification: **0 = Eye-open**, **1 = Eye-closed**) and examine the underlying signal characteristics, evaluate outlier mitigation strategies, investigate linear vs. non-linear separability, and test class differences using formal statistical hypothesis testing.

---

## 🔬 Key Highlights & Findings

- **Outlier Mitigation via IQR Clipping:** Raw EEG signals exhibited extreme amplitude spikes due to ocular and muscular artifacts (blinking, eye flutter, electrode movements), particularly in frontal electrodes. Clipping values using the Interquartile Range ($IQR$) preserved the full dataset size ($N=14,980$) and class balance while stabilizing downstream models.
- **Informative Electrodes:** Frontal and antero-frontal channels (`AF3`, `AF4`, `F4`, `F8`) showed the highest correlation with eye state and the highest Fisher Distance, reflecting proximity to ocular muscle activity.
- **PCA Variance vs. Class Separation Paradox:** High variance does **not** imply high class separability. While PC1 explained $52.7\%$ of total variance, its Fisher Distance was only $0.020$. In contrast, PC3 explained $12.8\%$ of variance but yielded a significantly higher Fisher Distance of $0.093$.
- **Supervised LDA vs. Unsupervised PCA:** Linear Discriminant Analysis (LDA) achieved superior class separation compared to PCA, increasing the best Fisher Distance from $0.093$ to $0.230$ and the silhouette score from $0.014$ to $0.079$.
- **Non-linear Boundaries & Class Overlap:** Clustering algorithms (K-Means, DBSCAN) and non-linear manifold embeddings (t-SNE, Self-Organizing Maps) revealed that the two eye states do not naturally form isolated geometric clusters (K-Means ARI = $0.017$, DBSCAN ARI = $0.003$), demonstrating significant non-linear overlap.
- **Statistically Significant Class Differences:** Welch's two-sample $t$-test on `AF3` confirmed a statistically significant mean difference across the full cleaned dataset ($t = -19.8745$, $p = 9.58 \times 10^{-87}$) with an effect size of Cohen's $d = -0.3255$.

---

## 📊 Dataset Summary

The dataset was obtained using an **Emotiv EPOC** neuroheadset across continuous measurements lasting $117$ seconds. The eye state was recorded simultaneously via a video camera and synchronized with the EEG stream.

| Property | Value |
| :--- | :--- |
| **Number of Samples** | 14,980 |
| **Number of Features** | 14 continuous EEG channels |
| **Target Variable** | `eye_state` |
| **Class 0** | Eye-open |
| **Class 1** | Eye-closed |
| **Missing Values** | 0 |
| **Sampling Mechanism** | 128 Hz (continuous time-series) |

### Electrode Mapping (Standard 10-20 System)
The 14 EEG channels were mapped to anatomically descriptive names:
- `AF3`: `antero_frontal_left_3`
- `F7`: `frontal_left_7`
- `F3`: `frontal_left_3`
- `FC5`: `frontocentral_left_5`
- `T7`: `temporal_left_7`
- `P7`: `parietal_left_7`
- `O1`: `occipital_left_1`
- `O2`: `occipital_right_2`
- `P8`: `parietal_right_8`
- `T8`: `temporal_right_8`
- `FC6`: `frontocentral_right_6`
- `F4`: `frontal_right_4`
- `F8`: `frontal_right_8`
- `AF4`: `antero_frontal_right_4`

---

## 🛠️ Methodology & Analytical Pipeline

```
Raw EEG Data (14,980 x 14)
          │
          ▼
1. Exploratory Data Analysis (EDA) & IQR Clipping (Outlier Handling)
          │
          ▼
2. Pearson Correlation Analysis (Inter-electrode & Label Dependencies)
          │
          ▼
3. Z-Score Standardization & Fisher Distance Quantification
          │
          ├─────────────────────────────┬─────────────────────────────┐
          ▼                             ▼                             ▼
4. Dimensionality Reduction     5. Clustering & Manifolds    6. Hypothesis Testing
   - PCA (Unsupervised)           - K-Means (PCA space)        - Welch's t-test (AF3)
   - LDA (Supervised)             - DBSCAN (PCA space)         - Cohen's d Effect Size
   - Variance vs Fisher Dist.     - t-SNE & SOM (Full space)   - Sample size stability
```

### 1. Outlier Detection and IQR Clipping
Due to involuntary blinks and micro-movements, raw signals exhibited extreme values that compressed the visual distribution in boxplots. Outliers were detected and clipped to bounds defined by:
$$\text{Lower Bound} = Q_1 - 1.5 \times IQR, \quad \text{Upper Bound} = Q_3 + 1.5 \times IQR$$
This eliminated extreme skewness without discarding valuable time-series samples.

### 2. Feature Scaling & Fisher Distance
Features were standardized via Z-score normalization ($z = \frac{x - \mu}{\sigma}$). To quantify how well individual features or projections discriminate between classes relative to within-class dispersion, the **Fisher Distance** was calculated:
$$FD = \frac{(\mu_0 - \mu_1)^2}{\sigma_0^2 + \sigma_1^2}$$

### 3. Dimensionality Reduction: PCA vs. LDA
- **Principal Component Analysis (PCA):** Orthogonal linear transformation maximizing variance ($Z = XW$).
- **Linear Discriminant Analysis (LDA):** Supervised linear projection maximizing the Fisher criterion:
  $$J(w) = \frac{w^T S_B w}{w^T S_W w}$$
  where $S_B$ is the between-class scatter matrix and $S_W$ is the within-class scatter matrix.

### 4. Clustering & Nonlinear Manifold Learning
- **K-Means & DBSCAN:** Evaluated on the first two principal components using Silhouette score, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).
- **t-SNE & Self-Organizing Maps (SOM):** Applied on the full normalized feature space to assess non-linear local neighborhood preservation and cluster dominance topology ($5 \times 5$ grid).

### 5. Hypothesis Testing
Welch's independent two-sample $t$-test evaluated the difference in mean activation of `antero_frontal_left_3` (`AF3`) between eye-open and eye-closed states:
$$t = \frac{\bar{x}_0 - \bar{x}_1}{\sqrt{\frac{s_0^2}{n_0} + \frac{s_1^2}{n_1}}}$$
Effect size was quantified using **Cohen's $d$**:
$$d = \frac{\bar{x}_0 - \bar{x}_1}{s_{\text{pooled}}}$$

---

## 📈 Quantitative Results

### Top Outliers and Informative Channels
| Feature | Outlier Count | Fisher Distance | Correlation with `eye_state` |
| :--- | :---: | :---: | :---: |
| `antero_frontal_right_4` (`AF4`) | **1,592** | **0.071** | **+0.184** |
| `antero_frontal_left_3` (`AF3`) | 1,485 | 0.053 | +0.160 |
| `frontal_right_8` (`F8`) | 1,460 | 0.041 | +0.141 |
| `frontal_right_4` (`F4`) | - | 0.043 | +0.144 |
| `occipital_left_1` (`O1`) | 348 | 0.004 | Subtle |
| `temporal_left_7` (`T7`) | - | 0.001 | Subtle |

### PCA vs. LDA Performance Comparison
| Metric | PCA (Unsupervised) | LDA (Supervised) |
| :--- | :---: | :---: |
| **Best Fisher Distance** | 0.093000 (PC3) | **0.230000** |
| **Silhouette Score** | 0.014459 | **0.078509** |

### Clustering Evaluation Metrics
| Method | ARI | NMI | Silhouette Score |
| :--- | :---: | :---: | :---: |
| **K-Means** | 0.0172 | 0.0086 | **0.4511** |
| **DBSCAN** (non-noise samples) | 0.0026 | 0.0014 | 0.3377 |

*Note: While K-Means achieves a moderate silhouette score due to geometric partitioning in 2D space, the near-zero ARI and NMI values demonstrate that unsupervised geometric clusters do not correspond to the true biological eye states.*

### Hypothesis Testing Results (`antero_frontal_left_3`)
| Experiment | Sample Size | $t$-statistic | $p$-value | Cohen's $d$ | Decision ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample 1** | 36 per class | -3.3020 | 0.0015 | -0.7780 | Reject $H_0$ |
| **Sample 2** | 64 per class | -1.9060 | 0.0590 | -0.3370 | Fail to Reject $H_0$ |
| **Full Cleaned Dataset** | **All 14,980** | **-19.8745** | **$9.58 \times 10^{-87}$** | **-0.3255** | **Reject $H_0$** |

---

## 📁 Repository Structure

```text
eeg_eye_state/
├── code.ipynb          # End-to-end Python / Jupyter notebook implementation
├── report.pdf          # Full academic project report (CSE555)
├── requirements.txt    # Python package dependencies
├── .gitignore          # Git exclusion rules
└── README.md           # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/VesileHann/eeg_eye_state.git
cd eeg_eye_state
```

### 3. Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Notebook
```bash
jupyter notebook code.ipynb
```
The notebook automatically downloads the dataset via `ucimlrepo` (ID: 264), performs all data cleaning, transformations, visualizations, and statistical analyses.

---

## 📖 Citation & References

1. **Roesler, O.** (2013). *EEG Eye State*. UCI Machine Learning Repository. [https://doi.org/10.24432/C57G7J](https://doi.org/10.24432/C57G7J)
2. **Delorme, A., Sejnowski, T., & Makeig, S.** (2007). *Enhanced Detection of Artifacts in EEG Data Using Higher-Order Statistics and Independent Component Analysis*. NeuroImage, 34(4), 1443–1449.
3. **Tukey, J. W.** (1977). *Exploratory Data Analysis*. Addison-Wesley.
4. **Fisher, R. A.** (1936). *The Use of Multiple Measurements in Taxonomic Problems*. Annals of Eugenics, 7(2), 179–188.
5. **van der Maaten, L., & Hinton, G.** (2008). *Visualizing Data Using t-SNE*. Journal of Machine Learning Research, 9, 2579–2605.
6. **Kohonen, T.** (1990). *The Self-Organizing Map*. Proceedings of the IEEE, 78(9), 1464–1480.
7. **Welch, B. L.** (1947). *The Generalization of Student’s Problem When Several Different Population Variances Are Involved*. Biometrika, 34(1–2), 28–35.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
