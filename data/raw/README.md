# Raw data

This project uses the **Kaggle Credit Card Fraud Detection** dataset. The CSV
is **not** committed to the repository (it is ~144 MB and covered by Kaggle's
terms). You must download it once and place it here.

## Expected file

```
data/raw/creditcard.csv
```

## Dataset facts

| Property            | Value                                             |
|---------------------|---------------------------------------------------|
| Rows                | 284,807 transactions                              |
| Fraud cases         | 492 (0.172%) — highly imbalanced                  |
| Features            | `Time`, `V1`..`V28` (PCA components), `Amount`    |
| Target              | `Class` (1 = fraud, 0 = legitimate)               |
| Source period       | Two days, September 2013, European cardholders    |

## How to download

### Option A — Website (no tooling required)
1. Sign in to Kaggle and open:
   <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
2. Click **Download** (you may need to accept the dataset terms).
3. Unzip and copy `creditcard.csv` into this folder.

### Option B — Kaggle CLI
```bash
pip install kaggle
# Place your API token at ~/.kaggle/kaggle.json (from Kaggle > Account > API)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

## Verify

From the project root:
```bash
python -c "from fraud_detection.data import load_raw_data; print(load_raw_data().shape)"
```
Expected output: `(284807, 31)`
