
# Birth & Fertility: Developed Countries — Analysis + Streamlit

This repo compares **birth rates** and **fertility rates** across developed countries and
offers an interactive **Streamlit** app to explore trends and run forecasts with three models:
**SARIMAX**, **ETS**, and **Prophet** (with optional covariates).

## Data (World Bank)
- `SP.DYN.CBRT.IN` — Crude birth rate (per 1,000 people)
- `SP.DYN.TFRT.IN` — Fertility rate, total (births per woman)
- `NY.GDP.PCAP.CD` — GDP per capita (current US$)
- `SL.TLF.CACT.FE.ZS` — Female labor force participation (% ages 15+)

## Quickstart

```bash
# Create env & install deps
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_birth_rates/app.py
```

### Notebook
See `notebooks/birth_rate_analysis_advanced.ipynb` for EDA and model comparison.

### Notes
- ETS doesn't use exogenous variables; SARIMAX and Prophet can include covariates.
- The app uses the World Bank API (pandas-datareader or wbdata). Cache CSVs if needed.
