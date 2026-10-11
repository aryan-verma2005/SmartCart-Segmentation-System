from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).parent / "smartcart_customers.csv"
SPENDING_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
REQUIRED_COLUMNS = {
    "ID",
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "Recency",
    *SPENDING_COLUMNS,
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Complain",
    "Response",
}
PALETTE = ["#a78bfa", "#34d399", "#fbbf24", "#fb7185", "#38bdf8", "#f97316", "#c084fc", "#2dd4bf"]
PLOT_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#cbd5e1", "family": "Inter, sans-serif"},
        "margin": {"l": 12, "r": 12, "t": 24, "b": 12},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "title": ""},
        "xaxis": {"gridcolor": "rgba(148,163,184,0.12)", "zerolinecolor": "rgba(148,163,184,0.12)"},
        "yaxis": {"gridcolor": "rgba(148,163,184,0.12)", "zerolinecolor": "rgba(148,163,184,0.12)"},
    }
}


st.set_page_config(
    page_title="SmartCart | Customer intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #eef2ff;
        --muted: #94a3b8;
        --surface: #111827;
        --line: rgba(148, 163, 184, .15);
        --mint: #34d399;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0b0f17; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(11, 15, 23, .82); }
    [data-testid="stSidebar"] {
        background: #0e1420;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.4rem; }
    .block-container { max-width: 1440px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
    h1 { font-size: clamp(2.3rem, 4vw, 3.35rem) !important; }
    h2 { font-size: 1.45rem !important; }
    p, label, [data-testid="stMarkdownContainer"] { color: #cbd5e1; }
    .brand {
        display: flex; align-items: center; gap: 11px; margin: 0 0 2rem;
        color: #f8fafc; font: 700 1.15rem 'Space Grotesk', sans-serif;
    }
    .brand-mark {
        display: grid; place-items: center; width: 38px; height: 38px; border-radius: 13px;
        background: linear-gradient(140deg, #34d399, #14b8a6); color: #06241d; font-size: 1.2rem;
        box-shadow: 0 8px 28px rgba(52, 211, 153, .2);
    }
    .eyebrow {
        color: var(--mint); font-size: .72rem; font-weight: 700; letter-spacing: .17em;
        text-transform: uppercase; margin-bottom: .65rem;
    }
    .hero-copy { color: var(--muted); max-width: 650px; font-size: 1.02rem; line-height: 1.7; }
    .hero-aside {
        padding: 18px 20px; border: 1px solid var(--line); border-radius: 18px;
        background: linear-gradient(145deg, rgba(52,211,153,.08), rgba(167,139,250,.07));
    }
    .hero-aside-label { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .1em; }
    .hero-aside-value { color: #f8fafc; font: 600 1.2rem 'Space Grotesk', sans-serif; margin-top: 6px; }
    [data-testid="stMetric"] {
        padding: 19px 20px; background: linear-gradient(145deg, #131b29, #101722);
        border: 1px solid var(--line); border-radius: 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
    [data-testid="stMetricValue"] { color: #f8fafc; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMetricDelta"] { color: var(--mint); }
    [data-testid="stTabs"] button { color: #94a3b8; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #34d399; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
    .section-note { color: var(--muted); font-size: .88rem; margin-top: -.6rem; }
    .stButton button, .stDownloadButton button {
        border: 1px solid rgba(52,211,153,.35); border-radius: 10px;
        background: rgba(52,211,153,.09); color: #a7f3d0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_customers(path: str) -> pd.DataFrame:
    customers = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(customers.columns))
    if missing_columns:
        raise ValueError(f"Customer data is missing required columns: {', '.join(missing_columns)}")

    customers["Income"] = customers["Income"].fillna(customers["Income"].median())
    customers["Dt_Customer"] = pd.to_datetime(customers["Dt_Customer"], dayfirst=True, errors="coerce")
    customers["Age"] = date.today().year - customers["Year_Birth"]
    customers["Customer_Tenure_Days"] = (
        customers["Dt_Customer"].max() - customers["Dt_Customer"]
    ).dt.days
    customers["Total_Spending"] = customers[SPENDING_COLUMNS].sum(axis=1)
    customers["Total_Children"] = customers["Kidhome"] + customers["Teenhome"]
    customers["Education_Group"] = customers["Education"].replace(
        {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate",
        }
    )
    customers["Household"] = customers["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Widow": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone",
        }
    )
    return customers.loc[(customers["Age"] < 90) & (customers["Income"] < 600_000)].copy()


@st.cache_data(show_spinner=False)
def segment_customers(customers: pd.DataFrame, cluster_count: int) -> tuple[pd.DataFrame, float, float]:
    numeric_features = [
        "Income",
        "Recency",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "Complain",
        "Response",
        "Age",
        "Customer_Tenure_Days",
        "Total_Spending",
        "Total_Children",
    ]
    categorical_features = pd.get_dummies(
        customers[["Education_Group", "Household"]].fillna("Unknown"),
        dtype=float,
    )
    features = pd.concat([customers[numeric_features], categorical_features], axis=1).fillna(0)
    scaled_features = StandardScaler().fit_transform(features)

    model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
    labels = model.fit_predict(scaled_features)
    silhouette = silhouette_score(scaled_features, labels)

    projection = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
    result = customers.copy()
    result["Cluster"] = labels
    result["PC1"] = projection[:, 0]
    result["PC2"] = projection[:, 1]
    variance_explained = PCA(n_components=2, random_state=42).fit(scaled_features).explained_variance_ratio_.sum()
    return result, float(silhouette), float(variance_explained)


def money(value: float) -> str:
    return f"${value:,.0f}"


def style_figure(figure: go.Figure, height: int = 380) -> go.Figure:
    figure.update_layout(**PLOT_TEMPLATE["layout"], height=height)
    return figure


st.sidebar.markdown(
    '<div class="brand"><span class="brand-mark">✳</span><span>smartcart</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Workspace")
st.sidebar.caption("Customer intelligence")
cluster_count = st.sidebar.slider("Customer segments", min_value=2, max_value=8, value=4)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model")
st.sidebar.markdown("**K-Means**")
st.sidebar.caption("Standardized customer behavior · PCA projection")
st.sidebar.markdown("---")
st.sidebar.caption("SmartCart · Retail analytics")

if not DATA_PATH.exists():
    st.error(f"Customer dataset not found at `{DATA_PATH}`.")
    st.stop()

try:
    customer_data = load_customers(str(DATA_PATH))
    customers, silhouette, variance_explained = segment_customers(customer_data, cluster_count)
except (OSError, ValueError) as error:
    st.error(f"Unable to prepare the customer segments: {error}")
    st.stop()

cluster_profile = (
    customers.groupby("Cluster")
    .agg(
        Customers=("ID", "size"),
        Avg_Spending=("Total_Spending", "mean"),
        Avg_Income=("Income", "mean"),
        Avg_Recency=("Recency", "mean"),
        Avg_Age=("Age", "mean"),
        Online_Orders=("NumWebPurchases", "mean"),
        Store_Orders=("NumStorePurchases", "mean"),
    )
    .reset_index()
)
cluster_profile["Share"] = cluster_profile["Customers"] / len(customers)
largest_cluster = cluster_profile.loc[cluster_profile["Customers"].idxmax()]

hero, hero_aside = st.columns([3.1, 1], vertical_alignment="center")
with hero:
    st.markdown('<div class="eyebrow">Retail, understood</div>', unsafe_allow_html=True)
    st.title("Know your customers.")
    st.markdown(
        '<p class="hero-copy">Turn purchase behavior into clear, actionable audience segments. '
        "Explore who shops with you, what they spend, and where to focus next.</p>",
        unsafe_allow_html=True,
    )
with hero_aside:
    st.markdown(
        f'<div class="hero-aside"><div class="hero-aside-label">Live model</div>'
        f'<div class="hero-aside-value">{cluster_count} customer segments</div>'
        f'<div class="hero-aside-label" style="margin-top:12px">Dataset refresh</div>'
        f'<div class="hero-aside-value">{customers["Dt_Customer"].max():%b %Y}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")
metric_columns = st.columns(4)
metric_columns[0].metric("Customers analyzed", f"{len(customers):,}")
metric_columns[1].metric("Average spend", money(customers["Total_Spending"].mean()))
metric_columns[2].metric("Largest segment", f"{int(largest_cluster['Customers']):,}")
metric_columns[3].metric("Silhouette score", f"{silhouette:.2f}")

overview_tab, segments_tab, customers_tab = st.tabs(["  Overview  ", "  Segment explorer  ", "  Customers  "])

with overview_tab:
    left, right = st.columns([1.75, 1], gap="large")
    with left:
        st.subheader("Spending & income")
        st.markdown('<p class="section-note">Each point represents a customer, colored by their assigned segment.</p>', unsafe_allow_html=True)
        scatter = px.scatter(
            customers,
            x="Income",
            y="Total_Spending",
            color=customers["Cluster"].astype(str),
            color_discrete_sequence=PALETTE[:cluster_count],
            hover_data={"ID": True, "Age": True, "Income": ":,.0f", "Total_Spending": ":,.0f", "Cluster": True},
            labels={"Income": "Annual income ($)", "Total_Spending": "Total spend ($)", "color": "Segment"},
            opacity=0.78,
        )
        scatter.update_traces(marker={"size": 8, "line": {"width": 0}})
        st.plotly_chart(style_figure(scatter, 430), width="stretch", config={"displayModeBar": False})

    with right:
        st.subheader("Segment mix")
        st.markdown('<p class="section-note">Share of the customer base by segment.</p>', unsafe_allow_html=True)
        mix = px.pie(
            cluster_profile,
            names="Cluster",
            values="Customers",
            hole=0.72,
            color="Cluster",
            color_discrete_sequence=PALETTE[:cluster_count],
        )
        mix.update_traces(
            textposition="outside",
            texttemplate="%{percent}",
            hovertemplate="Segment %{label}<br>%{value:,} customers<extra></extra>",
        )
        mix.add_annotation(
            text=f"<b>{len(customers):,}</b><br><span style='font-size:11px'>customers</span>",
            showarrow=False,
            font={"color": "#f8fafc", "size": 18},
        )
        st.plotly_chart(style_figure(mix, 430), width="stretch", config={"displayModeBar": False})

    st.subheader("Segment snapshot")
    st.markdown('<p class="section-note">Compare audience size, spending power, and recent activity.</p>', unsafe_allow_html=True)
    summary = cluster_profile.copy()
    summary["Avg_Spending"] = summary["Avg_Spending"].map(money)
    summary["Avg_Income"] = summary["Avg_Income"].map(money)
    summary["Share"] = summary["Share"].map(lambda share: f"{share:.0%}")
    summary["Avg_Recency"] = summary["Avg_Recency"].round(0).astype(int).astype(str) + " days"
    summary = summary.rename(
        columns={
            "Cluster": "Segment",
            "Customers": "Customers",
            "Share": "Audience share",
            "Avg_Spending": "Avg. spend",
            "Avg_Income": "Avg. income",
            "Avg_Recency": "Avg. days since purchase",
        }
    )[["Segment", "Customers", "Audience share", "Avg. spend", "Avg. income", "Avg. days since purchase"]]
    st.dataframe(summary, width="stretch", hide_index=True)

with segments_tab:
    selected_cluster = st.selectbox(
        "Explore a segment",
        options=cluster_profile["Cluster"].tolist(),
        format_func=lambda cluster: f"Segment {cluster + 1:02d}",
    )
    selected_profile = cluster_profile.loc[cluster_profile["Cluster"] == selected_cluster].iloc[0]
    selected_customers = customers.loc[customers["Cluster"] == selected_cluster]
    st.markdown(f"### Segment {selected_cluster + 1:02d}")
    profile_metrics = st.columns(4)
    profile_metrics[0].metric("Customers", f"{int(selected_profile['Customers']):,}")
    profile_metrics[1].metric("Audience share", f"{selected_profile['Share']:.0%}")
    profile_metrics[2].metric("Average spend", money(selected_profile["Avg_Spending"]))
    profile_metrics[3].metric("Average income", money(selected_profile["Avg_Income"]))

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Behavior profile")
        behavior = pd.DataFrame(
            {
                "Behavior": ["Online orders", "Store orders", "Catalog orders", "Web visits / month"],
                "Average": [
                    selected_profile["Online_Orders"],
                    selected_profile["Store_Orders"],
                    selected_customers["NumCatalogPurchases"].mean(),
                    selected_customers["NumWebVisitsMonth"].mean(),
                ],
            }
        )
        profile_chart = px.bar(
            behavior,
            x="Average",
            y="Behavior",
            orientation="h",
            color="Behavior",
            color_discrete_sequence=PALETTE,
            text_auto=".1f",
        )
        profile_chart.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_figure(profile_chart, 330), width="stretch", config={"displayModeBar": False})

    with right:
        st.subheader("Shopping preferences")
        category_spend = selected_customers[SPENDING_COLUMNS].mean().rename(
            {
                "MntWines": "Wine",
                "MntFruits": "Fruit",
                "MntMeatProducts": "Meat",
                "MntFishProducts": "Fish",
                "MntSweetProducts": "Sweets",
                "MntGoldProds": "Gold",
            }
        )
        preference = px.bar(
            x=category_spend.values,
            y=category_spend.index,
            orientation="h",
            color=category_spend.index,
            color_discrete_sequence=PALETTE,
            labels={"x": "Average spend ($)", "y": ""},
        )
        preference.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_figure(preference, 330), width="stretch", config={"displayModeBar": False})

    st.subheader("Where this segment sits")
    projection = px.scatter(
        customers,
        x="PC1",
        y="PC2",
        color=customers["Cluster"].astype(str),
        color_discrete_sequence=PALETTE[:cluster_count],
        opacity=0.22,
        labels={"PC1": "Principal component 1", "PC2": "Principal component 2", "color": "Segment"},
    )
    projection.add_trace(
        go.Scatter(
            x=selected_customers["PC1"],
            y=selected_customers["PC2"],
            mode="markers",
            name=f"Segment {selected_cluster + 1:02d}",
            marker={"color": PALETTE[selected_cluster % len(PALETTE)], "size": 8, "opacity": 0.86},
        )
    )
    projection.update_layout(legend={"title": ""})
    st.plotly_chart(style_figure(projection, 390), width="stretch", config={"displayModeBar": False})
    st.caption(f"The first two principal components explain {variance_explained:.0%} of feature variance.")

with customers_tab:
    st.subheader("Customer directory")
    st.markdown('<p class="section-note">Filter, inspect, and export your customer segments.</p>', unsafe_allow_html=True)
    filter_columns = st.columns([1, 1, 1.4])
    segment_options = sorted(customers["Cluster"].unique().tolist())
    selected_segments = filter_columns[0].multiselect(
        "Segments",
        options=segment_options,
        default=segment_options,
        format_func=lambda cluster: f"Segment {cluster + 1:02d}",
    )
    min_income, max_income = int(customers["Income"].min()), int(customers["Income"].max())
    income_range = filter_columns[1].slider(
        "Annual income",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income),
        format="$%d",
    )
    search = filter_columns[2].text_input("Find customer ID", placeholder="Type a customer ID")
    result_limit = st.selectbox("Rows to show", [25, 50, 100, 250], index=1)

    filtered = customers.loc[
        customers["Cluster"].isin(selected_segments)
        & customers["Income"].between(income_range[0], income_range[1])
    ].copy()
    if search.strip():
        filtered = filtered.loc[filtered["ID"].astype(str).str.contains(search.strip(), regex=False)]
    st.caption(f"{len(filtered):,} customers match your filters")

    visible = filtered[
        [
            "ID",
            "Cluster",
            "Age",
            "Education_Group",
            "Household",
            "Income",
            "Total_Spending",
            "NumWebPurchases",
            "NumStorePurchases",
            "Recency",
        ]
    ].rename(
        columns={
            "ID": "Customer ID",
            "Cluster": "Segment",
            "Age": "Age",
            "Education_Group": "Education",
            "Household": "Household",
            "Income": "Income ($)",
            "Total_Spending": "Total spend ($)",
            "NumWebPurchases": "Web orders",
            "NumStorePurchases": "Store orders",
            "Recency": "Days since purchase",
        }
    )
    st.dataframe(visible.head(result_limit), width="stretch", hide_index=True)
    st.download_button(
        "Download filtered customers",
        data=visible.to_csv(index=False).encode("utf-8"),
        file_name="smartcart_customer_segments.csv",
        mime="text/csv",
    )
