# SmartCart Segmentation System

An interactive customer intelligence dashboard for exploring retail customer
segments built from the included SmartCart customer dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads `smartcart_customers.csv` from the project folder. Adjust the
number of K-Means segments in the sidebar, compare customer spending and
income, inspect segment profiles, filter the customer directory, and download
the filtered results as a CSV.

## Segmentation

The dashboard follows the project notebook's feature engineering: customer
age, tenure, total product spending, household size, shopping channels,
recency, and grouped education/household categories. Missing income is filled
with the dataset median; customers aged 90 or older and incomes at or above
$600,000 are excluded. Features are standardized before K-Means clustering,
and PCA is used to visualize segment separation.

Dependencies are listed in `requirements.txt`.
