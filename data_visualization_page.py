import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Student Exam Data Visualization",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_dataset_10000_rows(1).csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("CSV file not found. Make sure 'student_dataset_10000_rows(1).csv' is in the same folder as this Python file.")
    st.stop()

# -----------------------------
# Title and explanation
# -----------------------------
st.title("📊 Student Exam Performance: Data Visualization")
st.write(
    "This page explores the main factors that may influence student exam performance, "
    "including study hours, attendance, sleep, internet usage, previous score, and assignments completed."
)
st.info("Note: The correlation heatmap is intentionally excluded because it is handled by another team member.")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filter Data")

placement_options = ["All"] + sorted(df["placement_status"].dropna().unique().tolist())
selected_placement = st.sidebar.selectbox("Placement Status", placement_options)

min_score = float(df["exam_score"].min())
max_score = float(df["exam_score"].max())
score_range = st.sidebar.slider(
    "Exam Score Range",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score)
)

filtered_df = df.copy()
if selected_placement != "All":
    filtered_df = filtered_df[filtered_df["placement_status"] == selected_placement]

filtered_df = filtered_df[
    (filtered_df["exam_score"] >= score_range[0]) &
    (filtered_df["exam_score"] <= score_range[1])
]

# -----------------------------
# KPI cards
# -----------------------------
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

avg_exam_score = filtered_df["exam_score"].mean()
avg_attendance = filtered_df["attendance"].mean()
avg_study_hours = filtered_df["study_hours"].mean()
placement_rate = (filtered_df["placement_status"].eq("Placed").mean()) * 100

col1.metric("Average Exam Score", f"{avg_exam_score:.2f}")
col2.metric("Average Attendance", f"{avg_attendance:.1f}%")
col3.metric("Average Study Hours", f"{avg_study_hours:.1f}")
col4.metric("Placement Rate", f"{placement_rate:.1f}%")

st.markdown("---")

# -----------------------------
# Dataset preview
# -----------------------------
with st.expander("View Dataset Preview"):
    st.dataframe(filtered_df.head(20))
    st.write(f"Rows shown after filters: {filtered_df.shape[0]}")

# -----------------------------
# Exam score distribution
# -----------------------------
st.subheader("1. Distribution of Exam Scores")
st.write("This chart shows how student exam scores are distributed overall.")

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(filtered_df["exam_score"], bins=25, kde=True, ax=ax)
ax.set_title("Distribution of Exam Scores")
ax.set_xlabel("Exam Score")
ax.set_ylabel("Number of Students")
st.pyplot(fig)

# -----------------------------
# Placement status count
# -----------------------------
st.subheader("2. Placement Status Count")
st.write("This chart compares how many students are placed versus not placed.")

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=filtered_df, x="placement_status", ax=ax)
ax.set_title("Number of Students by Placement Status")
ax.set_xlabel("Placement Status")
ax.set_ylabel("Number of Students")
st.pyplot(fig)

# -----------------------------
# Exam score by placement status
# -----------------------------
st.subheader("3. Exam Score by Placement Status")
st.write("This box plot compares exam scores between placed and not placed students.")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=filtered_df, x="placement_status", y="exam_score", ax=ax)
ax.set_title("Exam Score by Placement Status")
ax.set_xlabel("Placement Status")
ax.set_ylabel("Exam Score")
st.pyplot(fig)

# -----------------------------
# Study hours vs exam score
# -----------------------------
st.subheader("4. Study Hours vs Exam Score")
st.write("This scatter plot shows whether students who study more tend to receive higher exam scores.")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=filtered_df,
    x="study_hours",
    y="exam_score",
    hue="placement_status",
    alpha=0.6,
    ax=ax
)
ax.set_title("Study Hours vs Exam Score")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")
st.pyplot(fig)

# -----------------------------
# Attendance vs exam score
# -----------------------------
st.subheader("5. Attendance vs Exam Score")
st.write("This scatter plot shows the relationship between attendance and exam score.")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=filtered_df,
    x="attendance",
    y="exam_score",
    hue="placement_status",
    alpha=0.6,
    ax=ax
)
ax.set_title("Attendance vs Exam Score")
ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Exam Score")
st.pyplot(fig)

# -----------------------------
# Previous score vs exam score
# -----------------------------
st.subheader("6. Previous Score vs Exam Score")
st.write("This chart shows whether a student's previous score is related to their current exam score.")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=filtered_df,
    x="previous_score",
    y="exam_score",
    hue="placement_status",
    alpha=0.6,
    ax=ax
)
ax.set_title("Previous Score vs Exam Score")
ax.set_xlabel("Previous Score")
ax.set_ylabel("Exam Score")
st.pyplot(fig)

# -----------------------------
# Average exam score by assignment completion
# -----------------------------
st.subheader("7. Average Exam Score by Assignments Completed")
st.write("This bar chart shows how exam performance changes as students complete more assignments.")

assignment_summary = (
    filtered_df.groupby("assignments_completed", as_index=False)["exam_score"]
    .mean()
    .sort_values("assignments_completed")
)

fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=assignment_summary, x="assignments_completed", y="exam_score", ax=ax)
ax.set_title("Average Exam Score by Assignments Completed")
ax.set_xlabel("Assignments Completed")
ax.set_ylabel("Average Exam Score")
st.pyplot(fig)

# -----------------------------
# Sleep hours and exam score
# -----------------------------
st.subheader("8. Average Exam Score by Sleep Hours")
st.write("This chart explores whether students with different sleep hours show different exam performance.")

sleep_summary = (
    filtered_df.groupby("sleep_hours", as_index=False)["exam_score"]
    .mean()
    .sort_values("sleep_hours")
)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=sleep_summary, x="sleep_hours", y="exam_score", ax=ax)
ax.set_title("Average Exam Score by Sleep Hours")
ax.set_xlabel("Sleep Hours")
ax.set_ylabel("Average Exam Score")
st.pyplot(fig)

# -----------------------------
# Internet usage and exam score
# -----------------------------
st.subheader("9. Internet Usage vs Exam Score")
st.write("This chart helps examine whether higher internet usage is associated with higher or lower exam performance.")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=filtered_df,
    x="internet_usage",
    y="exam_score",
    hue="placement_status",
    alpha=0.6,
    ax=ax
)
ax.set_title("Internet Usage vs Exam Score")
ax.set_xlabel("Internet Usage")
ax.set_ylabel("Exam Score")
st.pyplot(fig)

# -----------------------------
# Main takeaway section
# -----------------------------
st.markdown("---")
st.subheader("Dashboard Takeaway")
st.write(
    "These visualizations help identify which student behaviors are connected with better exam performance. "
    "For example, the dashboard allows users to compare study hours, attendance, previous scores, sleep, "
    "and assignment completion with final exam outcomes. This supports the project objective of finding "
    "which adjustments may help students improve their results."
)
