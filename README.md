# Customer Segmentation Using K-Means and DBSCAN

## Overview
This project applies unsupervised machine learning to marketing campaign data to group customers with similar purchasing behaviour. It builds customer-level **Recency, Frequency, and Monetary (RFM)** features and compares K-Means with DBSCAN.

## Objectives
- Inspect the marketing campaign dataset.
- Engineer Recency, Frequency, and Monetary features.
- Log-transform Frequency and Monetary values and standardize the features.
- Segment customers using K-Means and DBSCAN.
- Evaluate K-Means using the silhouette score.
- Profile K-Means clusters with rule-based customer personas.
- Visualize segments using PCA and t-SNE.

## Dataset
The script expects `marketing_campaign.csv`, a tab-separated file despite its `.csv` extension.

The notebook version used in this project reports **2,240 rows and 29 columns**. Required columns include `ID`, `Recency`, `NumWebPurchases`, `NumCatalogPurchases`, and `NumStorePurchases`. Monetary value is calculated from all columns whose names begin with `Mnt`.

The dataset is not included. Obtain it from your authorized course/project source and place it beside `main.py`. Do not publish the data unless you have permission.

## Features
| Feature | Calculation |
|---|---|
| Recency | Existing `Recency` column |
| Frequency | Sum of web, catalog, and store purchases |
| Monetary | Sum of all `Mnt` spending columns |

`CustomerID` is retained for reference and is not included in the clustering features.

## Algorithms and workflow
1. Load the tab-separated dataset with pandas.
2. Build customer-level RFM features and remove rows with missing RFM values.
3. Apply `log1p` to Frequency and Monetary.
4. Standardize Recency, Frequency, and Monetary with `StandardScaler`.
5. Fit K-Means with 4 clusters, `n_init=10`, and `random_state=42`.
6. Estimate DBSCAN `eps` from the 85th percentile of fifth-nearest-neighbour distances; use `min_samples=5`.
7. Calculate the K-Means silhouette score.
8. Create cluster median profiles and heuristic personas.
9. Save PCA and t-SNE plots.

## Customer personas
| Persona | Suggested action |
|---|---|
| Champions / VIP | Offer loyalty rewards or referral perks |
| High-value at risk | Send a personalized win-back offer |
| Frequent buyers | Offer bundles or replenishment reminders |
| Dormant customers | Try a reactivation message |
| Occasional buyers | Send product recommendations |

Persona names are assigned using ordered rules based on each cluster's median RFM values relative to overall medians. They are heuristic interpretations and should be reviewed against the actual cluster profiles.

## Results from the original notebook
- Dataset shape: **2,240 × 29**
- DBSCAN `eps`: **0.195**
- K-Means silhouette score: **0.422**

These are the saved notebook's reported values; results can vary if the input dataset or environment differs.

## Project structure
```text
customer_segmentation_project/
├── main.py
├── CASESTUDY_3(CA_2).ipynb   # optional original notebook
├── marketing_campaign.csv    # add locally; not included
├── requirements.txt
├── .gitignore
├── README.md
└── outputs/                  # generated when the script runs
    ├── customer_segments.csv
    ├── customer_personas.csv
    ├── kmeans_pca.png
    └── tsne_kmeans_dbscan.png
```

## Setup and run

### 1. Create and activate a virtual environment (recommended)
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place `marketing_campaign.csv` in this project directory. If it is stored elsewhere, pass its path with `--data`.

### 4. Run the project
```bash
python main.py
```
Or specify paths:
```bash
python main.py --data path/to/marketing_campaign.csv --output outputs
```

The script saves customer-level cluster labels, persona profiles, and visualizations in the output directory.

## Project review

### Strengths
- Uses interpretable RFM-style customer features.
- Applies log transformation and feature scaling before clustering.
- Compares centroid-based K-Means and density-based DBSCAN.
- Uses reproducible random seeds for K-Means and t-SNE.
- Connects clusters to potential marketing actions.
- Produces PCA and t-SNE visualizations.

### Recommended improvements
1. Compare several K-Means values of `k` using an elbow plot and silhouette scores rather than assuming four clusters.
2. Evaluate DBSCAN separately by reporting the number of clusters (excluding noise) and the percentage of noise points.
3. Inspect outliers and missing data treatment before clustering.
4. Include cluster sizes and median RFM values in the final report.
5. Validate persona labels with campaign response, retention, or revenue outcomes.
6. Consider making DBSCAN parameters configurable and testing their sensitivity.

### Overall assessment
This is a useful exploratory segmentation project demonstrating feature engineering, preprocessing, clustering, dimensionality reduction, and business interpretation. The notebook's reported silhouette score of 0.422 suggests some cluster separation, but additional validation is needed before treating these segments as production-ready.

## Limitations
- This is an offline analysis script, not a deployed application or live segmentation service.
- Results depend on the dataset and preprocessing choices.
- K-Means is configured for four clusters in the current implementation.
- Persona labels are rule-based suggestions, not ground-truth customer categories.
- PCA and t-SNE plots are visual aids and do not independently prove cluster quality.

## Author
Add your name, course, institution, and project date here.

## License
Add a license before public redistribution. Confirm that you have permission to share the dataset and any third-party materials.
