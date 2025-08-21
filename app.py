
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data access
try:
    from pandas_datareader import wb as wbreader
    HAVE_WB_READER = True
except Exception:
    HAVE_WB_READER = False

try:
    import wbdata
    HAVE_WBDATA = True
except Exception:
    HAVE_WBDATA = False

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

st.set_page_config(page_title="Birth & Fertility Explorer", layout="wide")

IND_MAP = {
    "SP.DYN.CBRT.IN": "birth_rate",
    "SP.DYN.TFRT.IN": "fertility_rate",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "SL.TLF.CACT.FE.ZS": "female_lfp"
}
INDICATORS = list(IND_MAP.keys())

COUNTRIES = [
    "USA","CAN","GBR","DEU","FRA","ITA","ESP","NLD","BEL","LUX","IRL",
    "SWE","NOR","DNK","FIN","ISL","CHE","AUT","PRT","GRC",
    "JPN","KOR","AUS","NZL","CZE","SVK","SVN","POL","HUN","EST","LVA","LTU"
]

@st.cache_data(show_spinner=False)
def fetch_wb_panel(countries, indicators, start=1960, end=None):
    if end is None:
        end = pd.Timestamp.today().year - 1
    if HAVE_WB_READER:
        frames = []
        for ind in indicators:
            df = wbreader.download(indicator=ind, country=countries, start=start, end=end)
            df = df.reset_index().rename(columns={ind: IND_MAP[ind]})
            df["date"] = pd.to_datetime(df["year"].astype(int), format="%Y")
            df = df[["country","date", IND_MAP[ind]]]
            frames.append(df)
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on=["country","date"], how="outer")
        return out.sort_values(["country","date"]).reset_index(drop=True)
    if HAVE_WBDATA:
        frames = []
        for ind in indicators:
            df = wbdata.get_dataframe({ind: IND_MAP[ind]}, country=countries, convert_date=True).reset_index()
            df = df.rename(columns={ind: IND_MAP[ind]})
            df = df[["country","date", IND_MAP[ind]]]
            frames.append(df)
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on=["country","date"], how="outer")
        return out.sort_values(["country","date"]).reset_index(drop=True)
    raise ImportError("Please install pandas-datareader or wbdata.")

def zscore(s):
    s = pd.Series(s).astype(float)
    std = s.std(ddof=0)
    return (s - s.mean()) / (std if std != 0 else 1.0)

st.title("Birth & Fertility Explorer")
st.caption("World Bank data: Birth rate (target), Fertility rate, GDP per capita, Female LFP")

with st.sidebar:
    st.header("Controls")
    start = st.number_input("Start year", min_value=1950, max_value=2030, value=1960, step=1)
    end = st.number_input("End year (<= last complete year)", min_value=1961, max_value=2100, value=min(pd.Timestamp.today().year-1, 2024), step=1)
    country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("USA"))
    steps_ahead = st.slider("Forecast years ahead", min_value=3, max_value=20, value=10, step=1)
    use_covariates = st.checkbox("Use covariates (fertility, GDP pc, female LFP) where supported", value=True)
    model_name = st.selectbox("Model", ["SARIMAX","ETS","Prophet"])

# Fetch
panel = fetch_wb_panel(COUNTRIES, INDICATORS, start=int(start), end=int(end))
for col in ["birth_rate","fertility_rate","gdp_per_capita_usd","female_lfp"]:
    panel[col] = pd.to_numeric(panel[col], errors="coerce")

# EDA tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Country analysis", "Modeling & forecast"])

with tab1:
    st.subheader("Multi-country trends")
    pivot_birth = panel.pivot(index="date", columns="country", values="birth_rate").sort_index()
    pivot_fert  = panel.pivot(index="date", columns="country", values="fertility_rate").sort_index()

    fig, ax = plt.subplots()
    pivot_birth.plot(ax=ax)
    ax.set_title("Crude Birth Rate per 1,000 people")
    ax.set_xlabel("Year"); ax.set_ylabel("Birth rate")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    pivot_fert.plot(ax=ax)
    ax.set_title("Fertility Rate (births per woman)")
    ax.set_xlabel("Year"); ax.set_ylabel("Fertility rate")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)

with tab2:
    st.subheader(f"Country: {country}")
    g = panel[panel["country"]==country].dropna(subset=["birth_rate"]).sort_values("date").copy()
    g = g.set_index("date")
    fig, ax = plt.subplots()
    ax.plot(g.index, g["birth_rate"], label="Birth rate")
    if "fertility_rate" in g:
        ax.plot(g.index, g["fertility_rate"], label="Fertility rate")
    ax.set_title(f"{country} — birth vs fertility")
    ax.set_xlabel("Year"); ax.set_ylabel("Rate")
    ax.legend()
    st.pyplot(fig)

    latest_years = g.index.year.max()
    latest = g[g.index.year == latest_years]
    if not latest.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.scatter(g["gdp_per_capita_usd"], g["birth_rate"])
            ax.set_title(f"{country}: Birth rate vs GDP pc")
            ax.set_xlabel("GDP per capita (USD)"); ax.set_ylabel("Birth rate")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots()
            ax.scatter(g["female_lfp"], g["birth_rate"])
            ax.set_title(f"{country}: Birth rate vs Female LFP")
            ax.set_xlabel("Female LFP (%)"); ax.set_ylabel("Birth rate")
            st.pyplot(fig)

with tab3:
    st.subheader(f"Forecast — {model_name}")
    g = panel[panel["country"]==country].dropna(subset=["birth_rate"]).sort_values("date").copy()
    g = g.set_index("date").asfreq("Y")
    y = g["birth_rate"].dropna()

    exog_cols = ["fertility_rate","gdp_per_capita_usd","female_lfp"]
    X = g[exog_cols].apply(zscore)
    X = X.loc[y.index]

    test_years = 10 if len(y) > 20 else max(1, len(y)//5)
    train_y, test_y = y.iloc[:-test_years], y.iloc[-test_years:]
    train_X, test_X = X.loc[train_y.index], X.loc[test_y.index]

    if model_name == "SARIMAX":
        best_aic, best_order, best_model = np.inf, None, None
        for p in range(0,4):
            for d in range(0,3):
                for q in range(0,4):
                    try:
                        m = SARIMAX(train_y, exog=(train_X if use_covariates else None),
                                    order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
                        r = m.fit(disp=False)
                        if r.aic < best_aic:
                            best_aic, best_order, best_model = r.aic, (p,d,q), r
                    except Exception:
                        continue
        if best_model is not None:
            fc_test = best_model.get_forecast(steps=len(test_y), exog=(test_X if use_covariates else None)).predicted_mean
            # Refit full & forecast ahead
            full_m = SARIMAX(y, exog=(X if use_covariates else None), order=best_order,
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            future_index = pd.period_range(y.index[-1]+1, periods=steps_ahead, freq="Y")
            if use_covariates:
                last_row = X.iloc[[-1]].values
                future_X = pd.DataFrame(np.repeat(last_row, steps_ahead, axis=0), index=future_index, columns=X.columns)
            else:
                future_X = None
            fc_future = full_m.get_forecast(steps=steps_ahead, exog=future_X).predicted_mean
        else:
            fc_test, fc_future, best_order = pd.Series(index=test_y.index, dtype=float), pd.Series(dtype=float), None

        st.write("Selected order:", best_order)
        fig, ax = plt.subplots()
        ax.plot(train_y.index, train_y.values, label="Train")
        ax.plot(test_y.index, test_y.values, label="Test")
        if len(fc_test) > 0:
            ax.plot(fc_test.index, fc_test.values, linestyle="--", marker="o", label="Forecast (test)")
        ax.set_title(f"{country} — SARIMAX{best_order} test forecast")
        ax.set_xlabel("Year"); ax.set_ylabel("Birth rate (per 1,000)")
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(y.index, y.values, label="Historical")
        if len(fc_future) > 0:
            ax.plot(fc_future.index, fc_future.values, linestyle="--", marker="o", label=f"Forecast (+{steps_ahead}y)")
        ax.set_title(f"{country} — SARIMAX future forecast")
        ax.set_xlabel("Year"); ax.set_ylabel("Birth rate (per 1,000)")
        ax.legend()
        st.pyplot(fig)

    elif model_name == "ETS":
        try:
            ets = ExponentialSmoothing(train_y, trend="add", seasonal=None, initialization_method="estimated").fit()
            fc_test = ets.forecast(len(test_y))
            full_ets = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated").fit()
            fc_future = full_ets.forecast(steps_ahead)
        except Exception:
            fc_test, fc_future = pd.Series(index=test_y.index, dtype=float), pd.Series(dtype=float)

        fig, ax = plt.subplots()
        ax.plot(train_y.index, train_y.values, label="Train")
        ax.plot(test_y.index, test_y.values, label="Test")
        if len(fc_test) > 0:
            ax.plot(fc_test.index, fc_test.values, linestyle="--", marker="o", label="Forecast (test)")
        ax.set_title(f"{country} — ETS test forecast")
        ax.set_xlabel("Year"); ax.set_ylabel("Birth rate (per 1,000)")
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(y.index, y.values, label="Historical")
        if len(fc_future) > 0:
            ax.plot(fc_future.index, fc_future.values, linestyle="--", marker="o", label=f"Forecast (+{steps_ahead}y)")
        ax.set_title(f"{country} — ETS future forecast")
        ax.set_xlabel("Year"); ax.set_ylabel("Birth rate (per 1,000)")
        ax.legend()
        st.pyplot(fig)

    elif model_name == "Prophet":
        if HAVE_PROPHET:
            train_df = pd.DataFrame({"ds": train_y.index.to_timestamp(), "y": train_y.values})
            test_df  = pd.DataFrame({"ds": test_y.index.to_timestamp(), "y": test_y.values})
            if use_covariates:
                for c in X.columns:
                    train_df[c] = train_X[c].values
                    test_df[c]  = test_X[c].values

            m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            if use_covariates:
                for c in X.columns:
                    m.add_regressor(c)
            m.fit(train_df)

            future_test = test_df[["ds"] + (list(X.columns) if use_covariates else [])]
            fc_test_vals = m.predict(future_test)["yhat"].values
            fc_test = pd.Series(fc_test_vals, index=test_y.index)

            df_all = pd.DataFrame({"ds": y.index.to_timestamp(), "y": y.values})
            if use_covariates:
                for c in X.columns:
                    df_all[c] = X[c].values
            m2 = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            if use_covariates:
                for c in X.columns:
                    m2.add_regressor(c)
            m2.fit(df_all)
            future_dates = pd.period_range(y.index[-1]+1, periods=steps_ahead, freq="Y").to_timestamp()
            future = pd.DataFrame({"ds": future_dates})
            if use_covariates:
                last_vals = X.iloc[-1]
                for c in X.columns:
                    future[c] = float(last_vals[c])
            fc_future_vals = m2.predict(future)["yhat"].values
            fc_future = pd.Series(fc_future_vals, index=pd.period_range(y.index[-1]+1, periods=steps_ahead, freq="Y"))
        else:
            st.warning("Prophet is not installed. Add 'prophet' to requirements and reinstall.")
            fc_test, fc_future = pd.Series(index=test_y.index, dtype=float), pd.Series(dtype=float)

        fig, ax = plt.subplots()
        ax.plot(train_y.index, train_y.values, label="Train")
        ax.plot(test_y.index, test_y.values, label="Test")
        if len(fc_test) > 0:
            ax.plot(fc_test.index, fc_test.values, linestyle="--", marker="o", label="Forecast (test)")
        ax.set_title(f"{country} — Prophet test forecast")
        ax.set_xlabel("Year"); ax.set_ylabel("Birth rate (per 1,000)")
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(y.index, y.values, label="Historical")
        if len(fc_future) > 0:
            ax.plot(fc_future.index, fc_future.values, linestyle="--", marker="o", label=f"Forecast (+{steps_ahead}y)")
        ax.set_title(f"{country} — Prophet future forecast")
        ax.set_xlabel("Year"); ax.set_ylabel("Birth rate (per 1,000)")
        ax.legend()
        st.pyplot(fig)

st.info("Tip: for reproducibility, export the fetched panel to CSV and commit it to your repo.")
