import streamlit as st
import pandas as pd
from utils import build_feature_table, cluster_and_summary

st.set_page_config(page_title="Click & View Event Cohort Analysis", layout="wide")

st.title("Click & View Event Cohort Analysis")

st.markdown(
    """
Upload your **click events** and **view events** files.  
The app parses them, computes engagement metrics, runs k-means clustering,
and lets you download MLID cohorts.
"""
)

click_file = st.file_uploader("Click Events file (.xlsx)", type=["xlsx"])
view_file  = st.file_uploader("View Events file (.csv or .xlsx)", type=["csv", "xlsx"])

if click_file and view_file:
    with st.spinner("Reading files…"):
        click_df = pd.read_excel(click_file)
        view_df  = (
            pd.read_csv(view_file)
            if view_file.name.lower().endswith(".csv")
            else pd.read_excel(view_file)
        )
    st.success("Files loaded.")

    with st.spinner("Building features…"):
        feat_df = build_feature_table(click_df, view_df)
    st.write(f"Feature table shape: {feat_df.shape}")

    with st.spinner("Running k-means clustering…"):
        clustered, mean_summary, median_summary = cluster_and_summary(feat_df)
    st.success("Clustering complete!")

    st.subheader("Cluster Summary (mean)")
    st.dataframe(mean_summary)

    st.subheader("Cluster Summary (median)")
    st.dataframe(median_summary)

    st.subheader("Download MLIDs by cluster")
    for cl in sorted(clustered["cluster"].unique()):
        mlids = clustered[clustered["cluster"] == cl]["MLID"]
        st.download_button(
            f"Cluster {cl} MLIDs ({len(mlids)})",
            data=mlids.to_csv(index=False).encode(),
            file_name=f"cluster_{cl}_mlids.csv",
            mime="text/csv",
        )

    st.subheader("Download full feature table")
    st.download_button(
        "Full feature table (with clusters)",
        data=clustered.to_csv(index=False).encode(),
        file_name="features_with_clusters.csv",
        mime="text/csv",
    )
else:
    st.info("Upload both files to begin.")
