
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression


# Page setup

st.set_page_config(
    page_title="Student Exam Success Dashboard",
    page_icon="🎓",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# Load Data

@st.cache_data
def load_data():
    return pd.read_csv("student_dataset_10000_rows(1).csv")

# Converts Placed and Unplaced to Pass and fail
def clean_pass_fail_labels(df):
    df = df.copy()
    df["pass_fail"] = df["placement_status"].map({
        "Placed": "Passed",
        "Not Placed": "Failed"
    })
    return df


def add_pass_line(ax):
    ax.axhline(70, linestyle="--", linewidth=1.5)

def scatter_chart(df, x_col, x_label, title, explanation):
    """
    Creates a scatterplot of a student behavior against exam score,
    colored by pass/fail status.
    """
    st.subheader(title)
    st.caption(explanation)

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        data=df,
        x=x_col,
        y="exam_score",
        hue="pass_fail",
        alpha=0.65,
        ax=ax
    )

    add_pass_line(ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Exam Score")
    ax.set_title(title)
    ax.legend(title="Outcome")

    st.pyplot(fig)


# -----------------------------
# Load data
# -----------------------------
df = load_data()
df = clean_pass_fail_labels(df)

# Make sure expected columns exist
required_columns = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score",
    "exam_score",
    "pass_fail"
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎓 Dashboard Controls")
st.sidebar.write(
    "Use these filters to explore how different student groups perform."
)

min_score = float(df["exam_score"].min())
max_score = float(df["exam_score"].max())

score_range = st.sidebar.slider(
    "Filter by exam score",
    min_value=float(np.floor(min_score)),
    max_value=float(np.ceil(max_score)),
    value=(float(np.floor(min_score)), float(np.ceil(max_score)))
)

selected_outcomes = st.sidebar.multiselect(
    "Filter by outcome",
    options=sorted(df["pass_fail"].unique()),
    default=sorted(df["pass_fail"].unique())
)

filtered_df = df[
    (df["exam_score"] >= score_range[0]) &
    (df["exam_score"] <= score_range[1]) &
    (df["pass_fail"].isin(selected_outcomes))
]

# -----------------------------
# Header
# -----------------------------
st.title("🎓 Student Exam Success Dashboard")

st.markdown(
    """
    This dashboard explores the factors that influence whether students pass or fail an exam.
    Since passing is determined by scoring **70 or higher**, the goal is not only to show who passed,
    but to understand which student behaviors are associated with stronger exam performance.
    """
)

st.markdown("---")

# -----------------------------
# Section 1: Overall performance
# -----------------------------
st.header("1. Overall Student Performance")

st.markdown(
    """
    We begin with a high-level view of the class. These metrics summarize the general academic
    performance and give context before analyzing individual factors.
    """
)

total_students = len(filtered_df)
avg_score = filtered_df["exam_score"].mean()
pass_rate = (filtered_df["pass_fail"] == "Passed").mean() * 100
avg_study = filtered_df["study_hours"].mean()
avg_attendance = filtered_df["attendance"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Students", f"{total_students:,}")
col2.metric("Average Exam Score", f"{avg_score:.1f}")
col3.metric("Pass Rate", f"{pass_rate:.1f}%")
col4.metric("Average Attendance", f"{avg_attendance:.1f}%")

st.write("")

# Exam score distribution
st.subheader("Exam Score Distribution")
st.caption(
    "This chart shows how student scores are spread across the dataset. The dashed line marks the passing score of 70."
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(
    data=filtered_df,
    x="exam_score",
    bins=25,
    kde=True,
    ax=ax
)
ax.axvline(70, linestyle="--", linewidth=1.5)
ax.text(71, ax.get_ylim()[1] * 0.9, "Passing score: 70", fontsize=10)
ax.set_xlabel("Exam Score")
ax.set_ylabel("Number of Students")
ax.set_title("Distribution of Exam Scores")
st.pyplot(fig)


st.markdown("---")

# -----------------------------
# Section 2: Behavioral factors
# -----------------------------
st.header("2. Behavioral Factors Associated with Exam Performance")

st.markdown(
    """
    The next charts connect student behavior to exam score. Each point represents one student.
    The color shows whether the student passed or failed, making it easier to see how outcomes
    cluster around different habits.
    """
)

scatter_chart(
    filtered_df,
    "study_hours",
    "Study Hours",
    "Study Hours vs Exam Score",
    "This chart tests whether students who study more tend to score higher."
)

scatter_chart(
    filtered_df,
    "attendance",
    "Attendance (%)",
    "Attendance vs Exam Score",
    "This chart shows whether students with stronger attendance tend to perform better."
)

scatter_chart(
    filtered_df,
    "sleep_hours",
    "Sleep Hours",
    "Sleep Hours vs Exam Score",
    "This chart considers whether healthy sleep habits are connected with exam performance."
)

scatter_chart(
    filtered_df,
    "internet_usage",
    "Internet Usage",
    "Internet Usage vs Exam Score",
    "This chart checks whether higher internet usage is associated with lower or higher performance."
)




st.markdown("---")

# -----------------------------
# Section 3: Additional academic habits
# -----------------------------
st.header("3. Academic Preparation Factors")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Assignments Completed vs Exam Score")
    st.caption("This chart shows whether completing more assignments is associated with stronger exam outcomes.")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=filtered_df,
        x="assignments_completed",
        y="exam_score",
        hue="pass_fail",
        alpha=0.65,
        ax=ax
    )
    add_pass_line(ax)
    ax.set_xlabel("Assignments Completed")
    ax.set_ylabel("Exam Score")
    ax.set_title("Assignments Completed vs Exam Score")
    ax.legend(title="Outcome")
    st.pyplot(fig)

with col_b:
    st.subheader("Previous Score vs Exam Score")
    st.caption("This chart shows how strongly prior academic performance relates to current exam performance.")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=filtered_df,
        x="previous_score",
        y="exam_score",
        hue="pass_fail",
        alpha=0.65,
        ax=ax
    )
    add_pass_line(ax)
    ax.set_xlabel("Previous Score")
    ax.set_ylabel("Exam Score")
    ax.set_title("Previous Score vs Exam Score")
    ax.legend(title="Outcome")
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Section 4: Feature importance
# -----------------------------
st.header("4. Which Factors Matter Most?")

st.markdown(
    """
    To connect the visual analysis to prediction, we fit a simple Linear Regression model.
    The chart below shows which variables have the strongest positive or negative relationship
    with exam score in the model.
    """
)

feature_cols = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score"
]

X = df[feature_cols]
y = df["exam_score"]

model = LinearRegression()
model.fit(X, y)

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_
})

coef_df["Absolute Impact"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("Absolute Impact", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(coef_df["Feature"], coef_df["Coefficient"])
ax.axvline(0, linewidth=1)
ax.set_xlabel("Regression Coefficient")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance for Predicting Exam Score")
st.pyplot(fig)

st.success(
    "Final takeaway: The dashboard suggests that student success is connected to a combination of academic habits, classroom engagement, lifestyle factors, and previous performance. These insights can guide recommendations before moving into the prediction model."
)

# -----------------------------
# Section 5: Optional raw data preview
# -----------------------------
with st.expander("View Dataset Preview"):
    st.dataframe(filtered_df.head(50))
