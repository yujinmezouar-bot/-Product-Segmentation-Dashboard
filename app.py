import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    try:
        from hdbscan import HDBSCAN
        HDBSCAN_AVAILABLE = True
    except ImportError:
        HDBSCAN_AVAILABLE = False

st.set_page_config(page_title="Product Segmentation", layout="wide")

FEATURES = ['Number_of_Sales_Transactions', 'Total_Quantity_Sold', 'Total_Revenue',
            'Average_Sale_Value', 'Days_Since_Last_Sale', 'Product_Lifetime',
            'Average_Days_Between_Sales', 'Number_of_Distinct_Customers',
            'Average_VWAP', 'VWAP_Volatility', 'Return_Rate']

CLUSTER_METHODS = ["K-Means", "Hierarchical (Agglomerative)", "Gaussian Mixture Model", "DBSCAN", "HDBSCAN"]
DENSITY_METHODS = ("DBSCAN", "HDBSCAN")


def normalize_column_names(columns):
    return [str(c).strip().replace('.', '-') for c in columns]


@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = normalize_column_names(df.columns)
    return df


def build_product_features(df, snapshot_date=None):
    col_map = {
        'Oid': 'ProductID', 'Code': 'ProductCode', 'Label1': 'ProductName'
    }
    df = df.rename(columns=col_map)

    raw_dates = df['Date'].copy()
    df['Date'] = pd.to_datetime(raw_dates, format='%m/%d/%Y %H:%M', errors='coerce')

    still_bad = df['Date'].isna()
    if still_bad.any():
        df.loc[still_bad, 'Date'] = pd.to_datetime(raw_dates[still_bad], format='%m/%d/%Y', errors='coerce')

    still_bad = df['Date'].isna()
    if still_bad.any():
        df.loc[still_bad, 'Date'] = pd.to_datetime(raw_dates[still_bad], errors='coerce')

    still_bad = df['Date'].isna()
    if still_bad.any():
        numeric_candidates = pd.to_numeric(raw_dates[still_bad], errors='coerce')
        df.loc[still_bad, 'Date'] = pd.to_datetime(numeric_candidates, unit='D', origin='1899-12-30', errors='coerce')

    bad_dates = df['Date'].isna().sum()
    if bad_dates > 0:
        st.warning(f"{bad_dates} rows had invalid/unparseable dates and were dropped.")
        df = df.dropna(subset=['Date'])

    df['Type'] = pd.to_numeric(df['Type'], errors='coerce')
    valid_types = [10, 12]
    unknown_mask = ~df['Type'].isin(valid_types)
    n_unknown = unknown_mask.sum()
    if n_unknown > 0:
        st.warning(f"{n_unknown} rows had a Type other than 10 (sale) or 12 (return) and were dropped.")
        df = df[~unknown_mask]

    is_return = df['Type'] == 12
    df.loc[is_return, 'Quantity'] = -df.loc[is_return, 'Quantity'].abs()
    df.loc[is_return, 'Amount'] = -df.loc[is_return, 'Amount'].abs()
    df.loc[~is_return, 'Quantity'] = df.loc[~is_return, 'Quantity'].abs()
    df.loc[~is_return, 'Amount'] = df.loc[~is_return, 'Amount'].abs()

    if snapshot_date is None:
        snapshot_date = df['Date'].max() + pd.Timedelta(days=1)

    # sales-only subset for VWAP stats (returns shouldn't distort price behavior)
    sales_only = df[df['Type'] == 10]

    customer_col = 'Code-2' if 'Code-2' in df.columns else None

    agg_dict = {
        'ProductCode': ('ProductCode', 'first'),
        'ProductName': ('ProductName', 'first'),
        'Number_of_Sales_Transactions': ('Date', 'count'),
        'Total_Quantity_Sold': ('Quantity', 'sum'),
        'Total_Revenue': ('Amount', 'sum'),
        'First_Sale': ('Date', 'min'),
        'Last_Sale': ('Date', 'max'),
    }
    if customer_col:
        agg_dict['Number_of_Distinct_Customers'] = (customer_col, 'nunique')

    agg = df.groupby('ProductID').agg(**agg_dict).reset_index()

    if not customer_col:
        agg['Number_of_Distinct_Customers'] = np.nan

    agg['Average_Sale_Value'] = agg['Total_Revenue'] / agg['Number_of_Sales_Transactions']
    agg['Days_Since_Last_Sale'] = (snapshot_date - agg['Last_Sale']).dt.days
    agg['Product_Lifetime'] = (agg['Last_Sale'] - agg['First_Sale']).dt.days
    agg['Average_Days_Between_Sales'] = agg.apply(
        lambda r: r['Product_Lifetime'] / (r['Number_of_Sales_Transactions'] - 1)
        if r['Number_of_Sales_Transactions'] > 1 else np.nan,
        axis=1
    )

    # VWAP-based stats, computed on sales-only rows
    vwap_stats = sales_only.groupby('ProductID')['VWAP'].agg(['mean', 'std']).reset_index()
    vwap_stats.columns = ['ProductID', 'Average_VWAP', 'VWAP_Std']
    vwap_stats['VWAP_Volatility'] = vwap_stats['VWAP_Std'] / vwap_stats['Average_VWAP']
    agg = agg.merge(vwap_stats[['ProductID', 'Average_VWAP', 'VWAP_Volatility']], on='ProductID', how='left')

    # Return rate: |return amount| / total sale amount for that product
    return_amt = df[df['Type'] == 12].groupby('ProductID')['Amount'].sum().abs()
    sale_amt = df[df['Type'] == 10].groupby('ProductID')['Amount'].sum()
    return_rate = (return_amt / sale_amt).reindex(agg['ProductID']).fillna(0)
    agg['Return_Rate'] = return_rate.values

    return agg


def rfm_score(agg):
    agg['R_Score'] = pd.qcut(agg['Days_Since_Last_Sale'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
    agg['F_Score'] = pd.qcut(agg['Number_of_Sales_Transactions'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    agg['M_Score'] = pd.qcut(agg['Total_Revenue'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    agg['RFM_Score'] = agg['R_Score'] + agg['F_Score'] + agg['M_Score']
    return agg


def label_cluster(row):
    if row['RFM_Score'] >= 13 and row['Days_Since_Last_Sale'] < 30:
        return 'Star Products'
    elif row['F_Score'] >= 4 and row['Days_Since_Last_Sale'] < 60:
        return 'Steady Sellers'
    elif row['Product_Lifetime'] < 30:
        return 'New Products'
    elif row['Days_Since_Last_Sale'] > 180:
        return 'Dead Stock'
    elif row['Days_Since_Last_Sale'] > 90:
        return 'Declining Products'
    else:
        return 'Occasional Sellers'


def scale_features(agg):
    data = agg[FEATURES].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    return X


def compute_pca(X, n_components=3):
    pca = PCA(n_components=n_components)
    coords_all = pca.fit_transform(X)
    return coords_all, pca


def cluster_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model.fit_predict(X)


def cluster_agglomerative(X, k, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return model.fit_predict(X)


def cluster_gmm(X, k):
    model = GaussianMixture(n_components=k, random_state=42, n_init=5)
    return model.fit_predict(X)


def cluster_dbscan(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)


def cluster_hdbscan(X, min_cluster_size, min_samples=None):
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = model.fit_predict(X)
    probabilities = getattr(model, "probabilities_", None)
    if probabilities is None:
        probabilities = np.full(len(labels), np.nan)
    return labels, probabilities


def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    if mask.sum() < 2:
        return None
    if len(set(labels[mask])) < 2:
        return None
    try:
        return silhouette_score(X[mask], labels[mask])
    except Exception:
        return None


def run_method(X, method, k=None, eps=None, min_samples=None, min_cluster_size=None):
    probabilities = None
    if method == "K-Means":
        labels = cluster_kmeans(X, k)
    elif method == "Hierarchical (Agglomerative)":
        labels = cluster_agglomerative(X, k)
    elif method == "Gaussian Mixture Model":
        labels = cluster_gmm(X, k)
    elif method == "DBSCAN":
        labels = cluster_dbscan(X, eps, min_samples)
    elif method == "HDBSCAN":
        labels, probabilities = cluster_hdbscan(X, min_cluster_size, min_samples)
    else:
        raise ValueError(f"Unknown method: {method}")

    if probabilities is None:
        probabilities = np.full(len(labels), np.nan)

    n_noise = int((labels == -1).sum())
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters_found, n_noise, probabilities


def compute_k_distance(X, k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    return np.sort(distances[:, -1])


def dbscan_grid_search(X, eps_values, min_samples_values):
    rows = []
    n = len(X)
    for min_samples in min_samples_values:
        for eps in eps_values:
            labels = cluster_dbscan(X, eps, min_samples)
            n_noise = int((labels == -1).sum())
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            sil = safe_silhouette(X, labels)
            rows.append({
                "eps": eps, "min_samples": min_samples, "n_clusters": n_clusters,
                "noise_pct": n_noise / n * 100, "silhouette": sil
            })
    return pd.DataFrame(rows)


def kmeans_k_search(X, k_values):
    """Fit K-Means for each candidate K and score it by silhouette (and inertia for reference)."""
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        sil = safe_silhouette(X, labels)
        rows.append({
            "k": k,
            "silhouette": sil,
            "inertia": model.inertia_,
        })
    return pd.DataFrame(rows)


def get_pca_loadings_table(pca):
    n_components = pca.components_.shape[0]
    cols = [f'PC{i+1}' for i in range(n_components)]
    loadings = pd.DataFrame(pca.components_.T, columns=cols, index=FEATURES)
    for c in cols:
        loadings[f'{c}_abs'] = loadings[c].abs()
    return loadings


def interpret_pca(loadings, pca):
    def top_features(pc_col, n=3):
        return loadings[f'{pc_col}_abs'].sort_values(ascending=False).head(n).index.tolist()

    def direction_note(feature, pc_col):
        sign = "increases with" if loadings.loc[feature, pc_col] > 0 else "decreases with"
        return f"{feature} ({sign} {pc_col})"

    top_pc1 = top_features('PC1')
    top_pc2 = top_features('PC2')
    top_pc3 = top_features('PC3')

    pc1_desc = ", ".join(direction_note(f, 'PC1') for f in top_pc1)
    pc2_desc = ", ".join(direction_note(f, 'PC2') for f in top_pc2)
    pc3_desc = ", ".join(direction_note(f, 'PC3') for f in top_pc3)

    var1, var2, var3 = pca.explained_variance_ratio_[:3]
    cumulative_2 = var1 + var2
    cumulative_3 = var1 + var2 + var3

    text = f"""
**PC1** ({var1*100:.1f}% of variance) is mainly driven by: {pc1_desc}.
This axis mostly separates products by overall sales activity/value level.

**PC2** ({var2*100:.1f}% of variance) is mainly driven by: {pc2_desc}.
This axis mostly separates products along a secondary pattern independent of PC1
(e.g. recency/pricing behavior vs. sheer volume, depending on which features dominate above).

**PC3** ({var3*100:.1f}% of variance) is mainly driven by: {pc3_desc}.
This is a tertiary pattern, capturing whatever remaining structure isn't explained by PC1/PC2.

**Cumulative variance explained by PC1 + PC2: {cumulative_2*100:.1f}%**
**Cumulative variance explained by PC1 + PC2 + PC3: {cumulative_3*100:.1f}%**
"""
    return text, cumulative_2, cumulative_3


def get_top_products_per_component(agg, n=5):
    rows = []
    for pc in ['PC1', 'PC2', 'PC3']:
        top_high = agg.nlargest(n, pc)[['ProductCode', 'ProductName', pc]].copy()
        top_high['Component'] = pc
        top_high['End'] = 'Highest'
        top_high = top_high.rename(columns={pc: 'Score'})

        top_low = agg.nsmallest(n, pc)[['ProductCode', 'ProductName', pc]].copy()
        top_low['Component'] = pc
        top_low['End'] = 'Lowest'
        top_low = top_low.rename(columns={pc: 'Score'})

        rows.append(top_high)
        rows.append(top_low)

    result = pd.concat(rows, ignore_index=True)
    return result[['Component', 'End', 'ProductCode', 'ProductName', 'Score']]


def get_cluster_pca_profile(agg):
    profile = agg.groupby('Cluster')[['PC1', 'PC2', 'PC3']].mean().round(2)
    profile['Count'] = agg.groupby('Cluster').size()
    return profile.reset_index()


def scatter_by_labels(ax, coords, labels, title):
    is_noise = labels == -1
    if is_noise.any():
        ax.scatter(coords[is_noise, 0], coords[is_noise, 1], c='lightgray', alpha=0.5, s=15, label='Noise')
    real = ~is_noise
    scatter = ax.scatter(coords[real, 0], coords[real, 1], c=labels[real], cmap='tab10', alpha=0.75, s=15)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(title)
    return scatter


# ---------------- UI ----------------
st.title("📦 Product Segmentation Dashboard")
st.caption("Product segmentation using RFM-style behavior, price stability (VWAP), and return rate")

uploaded = st.file_uploader("Upload sales data file (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded:
    is_new_file = ('raw_df' not in st.session_state) or (st.session_state.get('file_name') != uploaded.name)

    if is_new_file:
        st.session_state.raw_df = load_data(uploaded)
        st.session_state.file_name = uploaded.name
        st.session_state.needs_recalc = True
        for key in ['agg', 'pca', 'active_method',
                    'compare_results', 'compare_methods', 'compare_pca', 'compare_coords',
                    'dbscan_grid_results',
                    '_apply_dbscan_best', '_pending_dbscan_eps', '_pending_dbscan_min_samples',
                    'kmeans_grid_results',
                    '_apply_kmeans_best_k', '_pending_kmeans_k']:
            st.session_state.pop(key, None)

    raw_df = st.session_state.raw_df
    st.success(f"Loaded {len(raw_df)} rows")

    required_cols = ['Oid', 'Code', 'Label1', 'Date', 'Quantity', 'Amount', 'Type', 'VWAP']
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    if 'Code-2' not in raw_df.columns:
        st.warning("No 'Code-2' (customer code) column found — Number_of_Distinct_Customers will be unavailable.")

    if 'agg' not in st.session_state or st.session_state.get('needs_recalc', True):
        with st.spinner("Computing product-level features..."):
            agg = build_product_features(raw_df)
            agg = rfm_score(agg)
        st.session_state.agg = agg
        st.session_state.needs_recalc = False

    agg = st.session_state.agg

    with st.expander("🐞 Debug info (data load & aggregation)", expanded=False):
        st.write(f"rows: {len(raw_df)}")
        st.write(f"unique Oid in raw data: {raw_df['Oid'].nunique()}")
        st.write(f"aggregated products (rows in `agg`): {len(agg)}")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clustering", "Segment Profiles", "Product Lookup"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Products", len(agg))
        c2.metric("Total Revenue", f"{agg['Total_Revenue'].sum():,.0f}")
        c3.metric("Avg Sale Value", f"{agg['Average_Sale_Value'].mean():,.1f}")
        c4.metric("Avg Return Rate", f"{agg['Return_Rate'].mean()*100:,.1f}%")
        st.dataframe(agg.head(50))

    with tab2:
        if not HDBSCAN_AVAILABLE:
            st.warning("HDBSCAN isn't available in this environment (needs scikit-learn >= 1.3, or the `hdbscan` "
                       "package). Other methods still work normally.")

        mode = st.radio("Mode", ["Explore one method", "Compare all methods"], horizontal=True)

        if len(agg) > 3000:
            st.caption("⚠️ Hierarchical clustering, DBSCAN and HDBSCAN can get slow on large product counts — this "
                       f"dataset has {len(agg)} products.")

        if mode == "Explore one method":
            available_methods = [m for m in CLUSTER_METHODS if m != "HDBSCAN" or HDBSCAN_AVAILABLE]
            method = st.selectbox("Clustering method", available_methods)

            k = None
            eps = None
            min_samples = None
            min_cluster_size = None
            if method in ("K-Means", "Hierarchical (Agglomerative)", "Gaussian Mixture Model"):
                if 'k_slider' not in st.session_state:
                    st.session_state.k_slider = 4

                if st.session_state.get('_apply_kmeans_best_k', False):
                    st.session_state.k_slider = st.session_state['_pending_kmeans_k']
                    st.session_state['_apply_kmeans_best_k'] = False

                k = st.slider("Number of Clusters (K)", 2, 10, key='k_slider')

                if method == "K-Means":
                    with st.expander("🔧 Auto-tune K (maximize Silhouette Score)"):
                        st.caption("Runs K-Means for each K in the chosen range and ranks them by silhouette "
                                   "score, so you can pick a well-separated K instead of guessing.")
                        k_range = st.slider("K range to try", 2, 10, (2, 10), key='kmeans_k_range')

                        if st.button("Run Silhouette Search", key='run_kmeans_search'):
                            if k_range[1] <= k_range[0]:
                                st.error("K range max must be greater than K range min.")
                            else:
                                X_tune = scale_features(agg)
                                k_values = list(range(k_range[0], k_range[1] + 1))
                                with st.spinner(f"Trying K = {k_range[0]}..{k_range[1]}..."):
                                    grid_df = kmeans_k_search(X_tune, k_values)
                                st.session_state.kmeans_grid_results = grid_df

                        if 'kmeans_grid_results' in st.session_state:
                            grid_df = st.session_state.kmeans_grid_results
                            valid = grid_df.dropna(subset=['silhouette'])
                            display_df = valid.sort_values('silhouette', ascending=False).reset_index(drop=True)

                            if len(display_df) == 0:
                                st.info("No K in this range produced a valid silhouette score — try widening the range.")
                            else:
                                st.dataframe(display_df.style.format({
                                    'silhouette': '{:.3f}', 'inertia': '{:.1f}'
                                }))
                                best = display_df.iloc[0]
                                st.caption(f"Best by silhouette: K={int(best['k'])}, "
                                           f"silhouette={best['silhouette']:.3f}")
                                if st.button("Apply Best K to Slider"):
                                    st.session_state['_pending_kmeans_k'] = int(best['k'])
                                    st.session_state['_apply_kmeans_best_k'] = True
                                    st.rerun()
            elif method == "DBSCAN":
                if 'dbscan_eps' not in st.session_state:
                    st.session_state.dbscan_eps = 1.0
                if 'dbscan_min_samples' not in st.session_state:
                    st.session_state.dbscan_min_samples = 5

                if st.session_state.get('_apply_dbscan_best', False):
                    st.session_state.dbscan_eps = st.session_state['_pending_dbscan_eps']
                    st.session_state.dbscan_min_samples = st.session_state['_pending_dbscan_min_samples']
                    st.session_state['_apply_dbscan_best'] = False

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    eps = st.slider("eps (neighborhood radius, in scaled-feature units)",
                                     0.1, 5.0, step=0.1, key='dbscan_eps')
                with dcol2:
                    min_samples = st.slider("min_samples", 2, 20, key='dbscan_min_samples')
                st.caption("DBSCAN doesn't take a target number of clusters — it finds dense regions itself and "
                           "labels sparse points as noise (-1). Tune eps/min_samples and re-run to shape the result, "
                           "or use the auto-tune tools below to get a starting point.")

                with st.expander("🔧 Auto-tune DBSCAN (k-distance plot + grid search)"):
                    X_tune = scale_features(agg)

                    st.markdown("**1. k-distance elbow plot**")
                    k_for_plot = st.slider("k (≈ min_samples) for k-distance plot", 2, 20,
                                            st.session_state.dbscan_min_samples, key='kdist_k')
                    k_distances = compute_k_distance(X_tune, k_for_plot)
                    fig_kd, ax_kd = plt.subplots(figsize=(7, 3.5))
                    ax_kd.plot(k_distances)
                    ax_kd.set_xlabel("Products, sorted by distance")
                    ax_kd.set_ylabel(f"Distance to {k_for_plot}-th nearest neighbor")
                    ax_kd.set_title("k-distance graph — read eps off the elbow")
                    ax_kd.grid(alpha=0.3)
                    st.pyplot(fig_kd)

                    st.divider()

                    st.markdown("**2. Grid search by silhouette score**")
                    gcol1, gcol2, gcol3 = st.columns(3)
                    with gcol1:
                        eps_min = st.number_input("eps min", 0.1, 5.0, 0.3, 0.1, key='grid_eps_min')
                    with gcol2:
                        eps_max = st.number_input("eps max", 0.1, 5.0, 2.0, 0.1, key='grid_eps_max')
                    with gcol3:
                        eps_step = st.number_input("eps step", 0.05, 1.0, 0.1, 0.05, key='grid_eps_step')
                    ms_range = st.slider("min_samples range to try", 2, 20, (3, 10), key='grid_ms_range')

                    if st.button("Run Grid Search"):
                        if eps_max <= eps_min:
                            st.error("eps max must be greater than eps min.")
                        else:
                            eps_values = np.round(np.arange(eps_min, eps_max + eps_step / 2, eps_step), 2)
                            ms_values = list(range(ms_range[0], ms_range[1] + 1))
                            with st.spinner(f"Trying {len(eps_values) * len(ms_values)} combinations..."):
                                grid_df = dbscan_grid_search(X_tune, eps_values, ms_values)
                            st.session_state.dbscan_grid_results = grid_df

                    if 'dbscan_grid_results' in st.session_state:
                        grid_df = st.session_state.dbscan_grid_results
                        valid = grid_df.dropna(subset=['silhouette'])
                        reasonable = valid[(valid['n_clusters'] >= 2) & (valid['noise_pct'] < 50)]
                        display_df = reasonable if len(reasonable) > 0 else valid
                        display_df = display_df.sort_values('silhouette', ascending=False).head(10).reset_index(drop=True)

                        if len(display_df) == 0:
                            st.info("No combination in this range produced valid clusters — try widening the eps range.")
                        else:
                            st.dataframe(display_df.style.format({
                                'eps': '{:.2f}', 'noise_pct': '{:.1f}%', 'silhouette': '{:.3f}'
                            }))
                            best = display_df.iloc[0]
                            st.caption(f"Best by silhouette: eps={best['eps']:.2f}, "
                                       f"min_samples={int(best['min_samples'])}, "
                                       f"{int(best['n_clusters'])} clusters, {best['noise_pct']:.1f}% noise.")
                            if st.button("Apply Best Combination to Sliders"):
                                st.session_state['_pending_dbscan_eps'] = float(best['eps'])
                                st.session_state['_pending_dbscan_min_samples'] = int(best['min_samples'])
                                st.session_state['_apply_dbscan_best'] = True
                                st.rerun()
            elif method == "HDBSCAN":
                min_cluster_size = st.slider(
                    "min_cluster_size (minimum products to count as a segment)", 5, 200, 30, 5
                )

            if st.button("Run Clustering"):
                X = scale_features(agg)
                coords_all, pca = compute_pca(X)
                labels, n_found, n_noise, probabilities = run_method(
                    X, method, k=k, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size
                )
                sil = safe_silhouette(X, labels)

                agg['Cluster'] = labels
                agg['Cluster_Probability'] = probabilities
                agg['PC1'] = coords_all[:, 0]
                agg['PC2'] = coords_all[:, 1]
                agg['PC3'] = coords_all[:, 2]
                st.session_state.agg = agg
                st.session_state.pca = pca
                st.session_state.active_method = method

                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Clusters Found", n_found)
                mcol2.metric("Silhouette Score", f"{sil:.3f}" if sil is not None else "n/a")
                if method in DENSITY_METHODS:
                    mcol3.metric("Noise Points", n_noise)
                if method == "HDBSCAN":
                    valid_probs = probabilities[~np.isnan(probabilities)]
                    if len(valid_probs) > 0:
                        st.caption(f"Average membership confidence: {valid_probs.mean()*100:.1f}%")

                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = scatter_by_labels(ax, coords_all[:, :2], np.asarray(labels), f"Product Distribution — {method}")
                legend = ax.legend(*scatter.legend_elements(), title="Cluster")
                ax.add_artist(legend)
                st.pyplot(fig)

                st.subheader("PCA Interpretation")
                loadings = get_pca_loadings_table(pca)
                interpretation_text, cumulative_2, cumulative_3 = interpret_pca(loadings, pca)
                st.markdown(interpretation_text)

                st.markdown("**Feature Contribution Table (PCA Loadings)**")
                st.dataframe(
                    loadings[['PC1', 'PC2', 'PC3']]
                    .style.background_gradient(cmap='coolwarm', axis=0)
                    .format("{:.3f}")
                )

                pcol1, pcol2 = st.columns(2)
                pcol1.metric("Cumulative Variance (PC1 + PC2)", f"{cumulative_2*100:.1f}%")
                pcol2.metric("Cumulative Variance (PC1 + PC2 + PC3)", f"{cumulative_3*100:.1f}%")

                st.markdown("**Cluster Profile on PCA Axes**")
                cluster_pca_profile = get_cluster_pca_profile(agg)
                st.dataframe(
                    cluster_pca_profile.style
                    .background_gradient(cmap='coolwarm', subset=['PC1', 'PC2', 'PC3'])
                    .format({'PC1': '{:.2f}', 'PC2': '{:.2f}', 'PC3': '{:.2f}'})
                )

                st.markdown("**Top Products per Component**")
                top_products = get_top_products_per_component(agg, n=5)
                st.dataframe(top_products.style.format({'Score': '{:.2f}'}))

                if method == "K-Means":
                    with st.expander("Elbow Method - Choosing the Best K"):
                        inertias = [KMeans(n_clusters=i, random_state=42, n_init=10).fit(X).inertia_ for i in range(2, 11)]
                        fig2, ax2 = plt.subplots()
                        ax2.plot(range(2, 11), inertias, marker='o')
                        ax2.set_xlabel("K"); ax2.set_ylabel("Inertia")
                        st.pyplot(fig2)

        else:
            st.caption("Runs K-Means, Hierarchical clustering, a Gaussian Mixture Model, DBSCAN, and HDBSCAN on the "
                       "same scaled features, then lines up their results side by side.")

            compare_methods = [m for m in CLUSTER_METHODS if m != "HDBSCAN" or HDBSCAN_AVAILABLE]

            k = st.slider("Number of Clusters (K) — used by K-Means, Hierarchical, and GMM", 2, 10, 4)
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                eps = st.slider("DBSCAN eps", 0.1, 5.0, 1.0, 0.1)
            with dcol2:
                min_samples = st.slider("DBSCAN min_samples", 2, 20, 5)

            min_cluster_size = None
            if HDBSCAN_AVAILABLE:
                min_cluster_size = st.slider(
                    "HDBSCAN min_cluster_size (minimum products to count as a segment)", 5, 200, 30, 5
                )

            if st.button("Run Comparison"):
                X = scale_features(agg)
                coords_all, pca = compute_pca(X)

                results = {}
                for method in compare_methods:
                    labels, n_found, n_noise, probabilities = run_method(
                        X, method, k=k, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size
                    )
                    sil = safe_silhouette(X, labels)
                    results[method] = {
                        "labels": np.asarray(labels), "n_found": n_found,
                        "n_noise": n_noise, "sil": sil, "probabilities": probabilities,
                    }
                st.session_state.compare_results = results
                st.session_state.compare_methods = compare_methods
                st.session_state.compare_pca = pca
                st.session_state.compare_coords = coords_all

            if 'compare_results' in st.session_state:
                results = st.session_state.compare_results
                compare_methods = st.session_state.compare_methods
                coords_all = st.session_state.compare_coords

                st.subheader("Comparison Summary")
                summary_rows = []
                for method, r in results.items():
                    summary_rows.append({
                        "Method": method,
                        "Clusters Found": r["n_found"],
                        "Noise Points": r["n_noise"] if method in DENSITY_METHODS else "—",
                        "Silhouette Score": f"{r['sil']:.3f}" if r["sil"] is not None else "n/a",
                    })
                st.dataframe(pd.DataFrame(summary_rows), hide_index=True)

                st.subheader("Cluster Assignments — Side by Side")
                n_methods = len(compare_methods)
                n_cols = 3 if n_methods > 4 else 2
                n_rows = int(np.ceil(n_methods / n_cols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5 * n_rows))
                axes_flat = np.atleast_1d(axes).flatten()
                for ax, method in zip(axes_flat, compare_methods):
                    labels = results[method]["labels"]
                    scatter_by_labels(ax, coords_all[:, :2], labels, method)
                for ax in axes_flat[len(compare_methods):]:
                    ax.axis('off')
                fig.tight_layout()
                st.pyplot(fig)

                st.subheader("How Much Do the Methods Agree?")
                ari_rows = []
                kmeans_labels = results["K-Means"]["labels"]
                for method in compare_methods:
                    if method == "K-Means":
                        continue
                    ari = adjusted_rand_score(kmeans_labels, results[method]["labels"])
                    ari_rows.append({"Method": method, "ARI vs. K-Means": f"{ari:.3f}"})
                st.dataframe(pd.DataFrame(ari_rows), hide_index=True)

                st.subheader("Use One Method's Results in Segment Profiles / Product Lookup")
                chosen = st.selectbox("Adopt clusters from:", compare_methods, key="adopt_method")
                if st.button("Apply Selected Method"):
                    agg['Cluster'] = results[chosen]["labels"]
                    agg['Cluster_Probability'] = results[chosen]["probabilities"]
                    agg['PC1'] = coords_all[:, 0]
                    agg['PC2'] = coords_all[:, 1]
                    agg['PC3'] = coords_all[:, 2]
                    st.session_state.agg = agg
                    st.session_state.pca = st.session_state.compare_pca
                    st.session_state.active_method = chosen
                    st.success(f"Segment Profiles and Product Lookup now use {chosen} clusters.")

    with tab3:
        if 'Cluster' in agg.columns:
            active_method = st.session_state.get('active_method', 'K-Means')
            st.caption(f"Showing profiles for: **{active_method}**")

            agg['Segment_Label'] = agg.apply(label_cluster, axis=1)
            profile_cols = FEATURES + ['PC1', 'PC2', 'PC3']
            profile = agg.groupby('Cluster')[profile_cols].mean().round(2)
            profile['Count'] = agg.groupby('Cluster').size()
            st.dataframe(profile)

            st.subheader("Segment Distribution (Business Labels)")
            seg_counts = agg['Segment_Label'].value_counts()
            fig3, ax3 = plt.subplots()
            ax3.bar(seg_counts.index, seg_counts.values, color='steelblue')
            plt.xticks(rotation=45)
            st.pyplot(fig3)

            st.subheader("Filter & Export Products")
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                cluster_options = sorted(agg['Cluster'].unique().tolist())
                selected_clusters = st.multiselect(
                    "Filter by Cluster", options=cluster_options, default=cluster_options
                )
            with fcol2:
                segment_options = sorted(agg['Segment_Label'].unique().tolist())
                selected_segments = st.multiselect(
                    "Filter by Segment Label", options=segment_options, default=segment_options
                )

            filtered = agg[
                agg['Cluster'].isin(selected_clusters) &
                agg['Segment_Label'].isin(selected_segments)
            ]

            st.caption(f"Showing {len(filtered)} of {len(agg)} products")
            display_cols = ['ProductCode', 'ProductName', 'Cluster', 'Segment_Label',
                             'Total_Revenue', 'Number_of_Sales_Transactions',
                             'Days_Since_Last_Sale', 'Return_Rate', 'Average_VWAP']
            if 'Cluster_Probability' in filtered.columns and filtered['Cluster_Probability'].notna().any():
                display_cols.append('Cluster_Probability')
            st.dataframe(filtered[display_cols])

            filtered_csv = filtered.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "Download Filtered Results (CSV)", filtered_csv,
                "product_segments_filtered.csv", "text/csv"
            )

            st.divider()
            csv = agg.to_csv(index=False).encode('utf-8-sig')
            st.download_button("Download All Results (CSV)", csv, "product_segments_all.csv", "text/csv")
        else:
            st.info("Run clustering first in the previous tab")

    with tab4:
        if 'Cluster' in agg.columns:
            search = st.text_input("Search product (name or code)")
            if search:
                res = agg[agg['ProductName'].str.contains(search, case=False, na=False) |
                          agg['ProductCode'].astype(str).str.contains(search, case=False, na=False)]
                if 'Cluster_Probability' in res.columns and res['Cluster_Probability'].notna().any():
                    st.caption("Cluster_Probability = confidence (0–1) that a product belongs to its assigned "
                               "cluster. Only meaningful for HDBSCAN; other methods show n/a.")
                st.dataframe(res)
        else:
            st.info("Run clustering first")
else:
    st.info("Please upload a data file to begin")
