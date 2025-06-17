import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF


def preprocess_data(df):
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df = numeric_df.fillna(numeric_df.mean())
    return numeric_df


def detect_anomalies(df, model):
    model.fit(df)
    predictions = model.predict(df)
    df_result = df.copy()
    df_result['anomaly'] = predictions
    return df_result


def plot_scatter(df, feature_x, feature_y):
    palette = {0: 'blue', 1: 'red'}
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='anomaly', palette=palette, ax=ax)
    plt.title("Scatter Plot: Normal vs Anomalies")
    st.pyplot(fig)


def plot_boxplot(df, feature):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y=feature, x='anomaly', palette='Set2', ax=ax)
    plt.title("Boxplot per Anomaly Class")
    st.pyplot(fig)


def get_model(name, contamination):
    if name == "IsolationForest":
        return IForest(contamination=contamination)
    elif name == "OneClassSVM":
        return OCSVM(contamination=contamination)
    elif name == "LocalOutlierFactor":
        return LOF(contamination=contamination)
    else:
        st.error("Unknown model selected.")
        return None


def main():
    st.set_page_config(page_title="Anomaly Detection App", layout="centered")
    st.title("🔍 Anomaly Detection Dashboard")

    uploaded_file = st.file_uploader("Upload CSV/Excel/JSON file", type=["csv", "xlsx", "json"])
    if uploaded_file:
        filetype = uploaded_file.name.split('.')[-1]
        if filetype == "csv":
            df = pd.read_csv(uploaded_file)
        elif filetype == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif filetype == "json":
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return

        st.write("### Raw Data Preview")
        st.dataframe(df.head())

        clean_df = preprocess_data(df)
        st.session_state['clean_df'] = clean_df

        st.write("### Cleaned Numeric Data")
        st.dataframe(clean_df.head())

        st.write("### Model Selection")
        model_name = st.selectbox("Choose model", ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"])
        contamination = st.slider("Contamination (anomaly proportion)", 0.01, 0.5, 0.1)

        if st.button("Run Detection"):
            model = get_model(model_name, contamination)
            result = detect_anomalies(clean_df, model)
            st.session_state['result_df'] = result
            st.success("Anomaly detection complete!")

    if 'result_df' in st.session_state:
        st.write("### Detection Result (with 'anomaly' column)")
        st.dataframe(st.session_state.result_df.head())

        features = list(st.session_state.result_df.columns)
        features.remove('anomaly')

        st.subheader("📈 Visualizations")

        x_feature = st.selectbox("Select X-axis Feature", features, key="x_feat")
        y_feature = st.selectbox("Select Y-axis Feature", features, key="y_feat")
        plot_scatter(st.session_state.result_df, x_feature, y_feature)

        box_feature = st.selectbox("Select Feature for Boxplot", features, key="box_feat")
        plot_boxplot(st.session_state.result_df, box_feature)


if __name__ == "__main__":
    main()