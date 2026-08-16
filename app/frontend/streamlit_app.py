from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="📊",
    layout="wide",
)

st.title("InsightFlow AI")
st.subheader("Bilingual Agentic Data Analytics Platform")

st.write(
    "Upload a dataset and describe your analytical problem "
    "in English, Myanmar, or mixed language."
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json", "txt"],
)

user_problem = st.text_area(
    "Describe your problem",
    placeholder=(
        "Example: Analyze monthly revenue trends, top products, "
        "and underperforming regions."
    ),
)

response_language = st.selectbox(
    "Response language",
    [
        "Same as question",
        "English",
        "မြန်မာ",
        "Bilingual",
    ],
)

if uploaded_file is not None:
    file_extension = Path(uploaded_file.name).suffix.lower()

    st.success(f"Uploaded: {uploaded_file.name}")

    try:
        if file_extension == ".csv":
            dataframe = pd.read_csv(uploaded_file)

        elif file_extension in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(uploaded_file)

        elif file_extension == ".json":
            dataframe = pd.read_json(uploaded_file)

        else:
            dataframe = None
            text_content = uploaded_file.getvalue().decode(
                "utf-8",
                errors="replace",
            )
            st.text_area(
                "Text preview",
                value=text_content[:5000],
                height=300,
            )

        if dataframe is not None:
            st.subheader("Dataset preview")

            col1, col2 = st.columns(2)
            col1.metric("Rows", f"{len(dataframe):,}")
            col2.metric("Columns", len(dataframe.columns))

            st.dataframe(
                dataframe.head(100),
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"Unable to read the uploaded file: {error}")

if st.button("Analyze", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a dataset.")

    elif not user_problem.strip():
        st.warning("Please describe the problem you want to analyze.")

    else:
        st.info(
            "The file was accepted. The analytics workflow "
            "will be implemented in the next development step."
        )

        st.write("Selected response language:", response_language)
        st.write("User problem:", user_problem)