# PATH FIX
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# IMPORTS
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import backend.profiler as profiler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="Data Monitoring Dashboard", page_icon="📊", layout="wide"
)

# THEME

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1,h2,h3 {color:#00FFD1 !important;}
[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.08);
    padding:15px;border-radius:12px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center;'>🚀 Data Monitoring Dashboard</h1>",
    unsafe_allow_html=True,
)
st.divider()

@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=50, max_depth=10) 
    model.fit(X, y)
    return model


# HELPERS
def show_table(df, h=350):
    st.dataframe(df, use_container_width=True, height=h)

@st.cache_data
def load_file(file):
    name = file.name.lower()    
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(file)
    elif name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    elif name.endswith(".txt"):
        return pd.read_csv(file)
    elif name.endswith(".pkl"):
        return pd.read_pickle(file)
    else:
        st.error("Unsupported file format")
        st.stop()


# SIDEBAR
st.sidebar.header("⚙️ Controls")
file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "txt", "pkl"],
)

# MAIN
if file:
    try:
        df = load_file(file)

        # OVERVIEW
        st.subheader("🔎 Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        q = profiler.calculate_data_quality_score(df)

        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Quality Score", f"{q['overall_score']:.1f}/100")
        c4.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum()/1024**2:.2f}")

        with st.expander("👀 Preview Dataset"):
            show_table(df.head(50))

        report = profiler.check_data_quality(df)

        # TABS (added Model Evaluation)
        tabs = st.tabs(
            [
                "💡 Recommendations",
                "📊 Quality",
                "📈 Statistics",
                "📊 Correlation",
                "⚠️ Outliers",
                "🤖 Anomalies",
                "🎯 Cardinality",
                "💾 Memory",
                "🧪 Model Evaluation",
            ]
        )

        # RECOMMENDATIONS
        with tabs[0]:
            recs = profiler.generate_recommendations(df, report)
            for i, r in enumerate(recs, 1):
                st.write(f"**{i}. {r['type']}**")
                st.write("Issue:", r["issue"])
                st.write("Fix:", r["solution"])
                st.divider()

        # QUALITY
        with tabs[1]:
            st.plotly_chart(
                profiler.plot_null_distribution(df), use_container_width=True
            )
            st.plotly_chart(profiler.plot_null_heatmap(df), use_container_width=True)
            st.plotly_chart(
                profiler.plot_duplicate_analysis(df), use_container_width=True
            )

        # STATISTICS
        with tabs[2]:
            stats = profiler.get_statistical_summary(df)
            if stats:
                stats_df = pd.DataFrame(stats)
                show_table(stats_df)
                fig = profiler.plot_statistical_summary(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # CORRELATION
        with tabs[3]:
            fig = profiler.plot_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                rel = profiler.analyze_column_relationships(df)
                if rel:
                    show_table(
                        pd.DataFrame(rel.items(), columns=["Columns", "Correlation"])
                    )

        # OUTLIERS
        with tabs[4]:
            out = profiler.detect_outliers_iqr(df)
            if out:
                show_table(pd.DataFrame(out).T)

            nums = df.select_dtypes(include="number").columns
            if len(nums):
                col = st.selectbox("Column", nums)
                st.plotly_chart(
                    profiler.plot_outliers(df, col), use_container_width=True
                )

        # ANOMALIES
        with tabs[5]:
            an = profiler.detect_anomalies_isolation_forest(df)
            if an:
                st.metric("Total Anomalies", an["total_anomalies"])
                fig = profiler.plot_anomalies(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # CARDINALITY
        with tabs[6]:
            card = profiler.analyze_cardinality(df)
            if card:
                show_table(
                    pd.DataFrame(card.items(), columns=["Column", "Unique Values"])
                )
                st.plotly_chart(profiler.plot_cardinality(df), use_container_width=True)

        # MEMORY
        with tabs[7]:
            mem = profiler.analyze_memory_usage(df)
            st.metric("Total Memory (MB)", f"{mem['total_memory_mb']:.2f}")
            st.plotly_chart(profiler.plot_memory_usage(df), use_container_width=True)

        # MODEL EVALUATION
        with tabs[8]:
            st.subheader("🧪 Precision • Recall • F1 Score")
            cols = df.columns.tolist()
            target = st.selectbox("Select Target Column", cols, key="target_col")

            if target:
                X = df.drop(columns=[target])
                y = df[target]

                # Use numeric columns only
                X = X.select_dtypes(include="number").fillna(0)

                if len(X.columns) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )

                    model = train_model(X_train, y_train)
                    preds = model.predict(X_test)

                    # Classification report
                    report = classification_report(y_test, preds, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    class_report_df = (
                        report_df.iloc[:-3]
                        .reset_index()
                        .rename(columns={"index": "Class"})
                    )

                    st.success("✅ Model trained successfully")

                    # 🟢 UI Enhancement: Show table and chart side by side
                    t1, t2 = st.columns([1, 2])
                    with t1:
                        st.markdown("### Class-wise Metrics")
                        st.dataframe(class_report_df, use_container_width=True)

                    with t2:
                        # Metric selector
                        metrics = st.multiselect(
                            "Select metrics to plot",
                            ["precision", "recall", "f1-score"],
                            default=["precision", "recall", "f1-score"],
                        )

                        if metrics:
                            fig = px.bar(
                                class_report_df,
                                x="Class",
                                y=metrics,
                                barmode="group",
                                title="Class-wise Metrics",
                                text_auto=".2f",
                                color_discrete_sequence=px.colors.qualitative.D3,
                            )
                            fig.update_layout(
                                yaxis_title="Score",
                                xaxis_title="Class",
                                legend_title="Metric",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # ✅ Overall metrics
                    st.markdown("### Overall Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{report['accuracy']:.2f}")
                    col2.metric("Macro F1", f"{report['macro avg']['f1-score']:.2f}")
                    col3.metric(
                        "Weighted F1", f"{report['weighted avg']['f1-score']:.2f}"
                    )

                else:
                    st.warning("Need numeric columns for model training.")
    except Exception as e:
        import traceback

        st.error(str(e))
        st.text_area("Traceback", traceback.format_exc(), height=300)

else:
    st.info("⬅️ Upload a dataset to begin.")
