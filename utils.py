import pandas as pd
import numpy as np
import re, ast, json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------- regex patterns for event categories ----------
CLICK_PATTERNS = {
    "price_sheet":       re.compile(r"PRICE\\$(UNLOCK_PRICESHEET|PRICE_SHEET)", re.I),
    "brochure_download": re.compile(r"BROCHURE_DOWNLOAD", re.I),
    "plan_click":        re.compile(r"PLANS\\$", re.I),
    "video_click":       re.compile(r"VIDEOS\\$", re.I),
    "payment":           re.compile(r"PAYMENT_STRUCTURE|EMI_CALCULATOR", re.I),
    "whatsapp":          re.compile(r"WHATSAPP", re.I),
    "contact_or_otp":    re.compile(r"ENQUIRY_DIALOG\\$(GET_OTP|SUBMIT_LEAD|LEAD_CREATED)", re.I),
}

VIEW_PATTERNS = {
    "price_view":    re.compile(r"PRICE_", re.I),
    "brochure_view": re.compile(r"BROCHURE", re.I),
    "plan_view":     re.compile(r"PLAN", re.I),
    "video_view":    re.compile(r"VIDEO", re.I),
    "gallery_view":  re.compile(r"GALLERY", re.I),
}

# ---------- helpers ----------
def _safe_literal_eval(s):
    if not isinstance(s, str):
        return []
    s_clean = s.replace("\\n", " ")
    try:
        return ast.literal_eval(s_clean)
    except Exception:
        # try JSON style
        try:
            return json.loads(s_clean.replace("'", '"'))
        except Exception:
            return []

# ---------- feature builders ----------
def parse_click_df(df, click_col="Click Events LOFT"):
    df = df.copy()
    df["events_list"] = df[click_col].apply(_safe_literal_eval)
    for name, pat in CLICK_PATTERNS.items():
        df[name] = df["events_list"].apply(
            lambda lst: sum(1 for e in lst if pat.search(e))
        )
    df["n_clicks"]       = df["events_list"].apply(len)
    df["n_click_unique"] = df["events_list"].apply(lambda lst: len(set(lst)))
    keep = ["MLID"] + list(CLICK_PATTERNS.keys()) + ["n_clicks", "n_click_unique"]
    return df[keep]

def parse_view_df(df, view_col="View Events"):
    df = df.copy()
    df["events_list"] = df[view_col].apply(_safe_literal_eval)
    for name, pat in VIEW_PATTERNS.items():
        df[name] = df["events_list"].apply(
            lambda lst: sum(1 for e in lst if isinstance(e, list) and pat.search(e[0]))
        )
    df["n_view_events"] = df["events_list"].apply(len)
    df["n_view_unique"] = df["events_list"].apply(
        lambda lst: len(set(e[0] for e in lst if isinstance(e, list)))
    )
    df["total_view_time"] = df["events_list"].apply(
        lambda lst: sum(float(e[1]) for e in lst if isinstance(e, list) and len(e) > 1)
    )
    keep = ["MLID"] + list(VIEW_PATTERNS.keys()) + [
        "n_view_events",
        "n_view_unique",
        "total_view_time",
    ]
    return df[keep]

def build_feature_table(click_df, view_df,
                        click_col="Click Events LOFT",
                        view_col="View Events"):
    click_feats = parse_click_df(click_df, click_col)
    view_feats  = parse_view_df(view_df, view_col)
    return pd.merge(click_feats, view_feats, on="MLID", how="inner")

def cluster_and_summary(features_df, n_clusters=3, random_state=42):
    feat_cols = [c for c in features_df.columns if c != "MLID"]
    X_scaled  = StandardScaler().fit_transform(features_df[feat_cols])

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    features_df["cluster"] = km.fit_predict(X_scaled)

    summary_mean   = features_df.groupby("cluster")[feat_cols].mean().reset_index()
    summary_median = features_df.groupby("cluster")[feat_cols].median().reset_index()
    return features_df, summary_mean, summary_median
