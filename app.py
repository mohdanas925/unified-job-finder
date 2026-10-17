
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

from src import utils


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "notebook", "data", "Prediction.csv"
)

VECTORIZER_PATH = os.path.join(BASE_DIR, "notebook", "cv.pkl")
VECTORS_PATH = os.path.join(BASE_DIR, "notebook", "vectors.pkl")


# --------------------------------------------------
# 2. LOAD SAVED FILES
# --------------------------------------------------

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "title",
        "locations_derived",
        "url",
        "normalized_skills",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    return df


@st.cache_resource
def load_vectorizer():
    with open(VECTORIZER_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_job_vectors():
    with open(VECTORS_PATH, "rb") as file:
        return pickle.load(file)


# --------------------------------------------------
# 3. EXTRACT RESUME TEXT
# --------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    if not text.strip():
        raise ValueError(
            "No readable text found in the PDF. "
            "A scanned resume may require OCR."
        )

    return text


# --------------------------------------------------
# 4. RECOMMEND TOP MATCHING JOBS
# --------------------------------------------------

def recommend_jobs(resume_text, df, cv, vectors, top_n=10):

    # Use your existing preprocessing functions
    cleaned_text = utils.clean_text(resume_text)
    cleaned_text = utils.remove_stopwords(cleaned_text)

    # Your extract_skills function accepts only one argument
    raw_skills = utils.extract_skills(cleaned_text)

    if not raw_skills:
        raise ValueError(
            "No skills were extracted from the resume. "
            "Check the NER model output and entity labels."
        )

    # Your clean_skills function expects a string representation
    # of a Python list
    cleaned_skills = utils.clean_skills(str(raw_skills))

    # Apply your existing synonym normalization
    normalized_skills = utils.normalize_skills(cleaned_skills)

    if not normalized_skills:
        raise ValueError(
            "No skills remained after cleaning and normalization."
        )

    # Convert skills into the format expected by CountVectorizer
    resume_skills_text = " ".join(normalized_skills)

    resume_vector = cv.transform([resume_skills_text])

    if resume_vector.nnz == 0:
        raise ValueError(
            "None of the extracted skills matched the CountVectorizer "
            "vocabulary. Check your preprocessing and saved cv.pkl."
        )

    # Ensure the vectors correspond to the loaded dataset
    if vectors.shape[0] != len(df):
        raise ValueError(
            f"Dataset has {len(df)} rows, but vectors.pkl has "
            f"{vectors.shape[0]} rows. Load the exact dataset used "
            "to generate vectors.pkl, in the same row order."
        )

    # Calculate cosine similarity
    scores = cosine_similarity(
        resume_vector,
        vectors
    ).ravel()

    # Select the 10 highest scores
    top_indices = np.argsort(scores)[::-1][:top_n]

    recommendations = df.iloc[top_indices].copy()
    recommendations["similarity_score"] = (
        scores[top_indices] * 100
    ).round(2)

    return raw_skills, normalized_skills, recommendations[
        [
            "title",
            "locations_derived",
            "similarity_score",
            "url",
            "normalized_skills",
        ]
    ].reset_index(drop=True)


# --------------------------------------------------
# 5. STREAMLIT INTERFACE
# --------------------------------------------------

st.set_page_config(
    page_title="Unified Job Finder",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Unified Job Finder")
st.write(
    "Upload your resume in PDF format to find the "
    "10 most similar jobs based on your skills."
)

uploaded_resume = st.file_uploader(
    "Upload your resume",
    type=["pdf"],
)

if st.button("Find Top 10 Jobs", type="primary"):

    if uploaded_resume is None:
        st.warning("Please upload your resume first.")

    else:
        try:
            with st.spinner("Loading dataset and saved models..."):
                df = load_dataset()
                cv = load_vectorizer()
                vectors = load_job_vectors()

            with st.spinner("Extracting and matching your skills..."):
                resume_text = extract_text_from_pdf(uploaded_resume)

                raw_skills, normalized_skills, recommendations = (
                    recommend_jobs(
                        resume_text=resume_text,
                        df=df,
                        cv=cv,
                        vectors=vectors,
                        top_n=10,
                    )
                )

            st.success("Recommendations generated successfully!")

            st.subheader("Skills extracted from your resume")
            st.write(", ".join(raw_skills))

            st.subheader("Normalized skills used for matching")
            st.write(", ".join(normalized_skills))

            st.subheader("Top 10 matching jobs")

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "similarity_score": st.column_config.NumberColumn(
                        "Similarity Score (%)",
                        format="%.2f%%",
                    ),
                    "url": st.column_config.LinkColumn(
                        "Job Link",
                        display_text="Open job",
                    ),
                },
            )

            csv_data = recommendations.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download recommendations as CSV",
                data=csv_data,
                file_name="job_recommendations.csv",
                mime="text/csv",
            )

            st.caption(
                "Similarity scores indicate vector similarity, "
                "not the percentage of job requirements satisfied."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
