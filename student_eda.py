import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create student performance dataset

data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Gender": [
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Female"
    ],
    "StudyHours": [
        5, 3, 6, 4, 7, 2, 8, 5, 4, 6, 7, 5
    ],
    "Attendance(%)": [
        90, 82, 95, 78, 92, 75, 96, 85, 88, 91, 89, 93
    ],
    "Marks": [
        82, 68, 88, np.nan, 91, 60,
        95, 72, np.nan, 80, 86, 82
    ]
}

df_original = pd.DataFrame(data)

# Add one exact duplicate row
df_original = pd.concat(
    [df_original, df_original.iloc[[0]]],
    ignore_index=True
)

# Save dataset
df_original.to_csv("student_performance.csv", index=False)


# 2. Load CSV

df = pd.read_csv("student_performance.csv")

print("Shape:", df.shape)

print("\nDataset Information:")
df.info()


# 3. Missing values and duplicates

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# 4. Clean the data

# Remove duplicate rows
df_clean = df.drop_duplicates()

# Fill missing Marks with median
marks_median = df_clean["Marks"].median()

df_clean["Marks"] = df_clean["Marks"].fillna(marks_median)

print("\nMedian Marks:", marks_median)

print("\nAfter cleaning:", df_clean.shape)


# 5. Histogram of Marks

plt.figure(figsize=(7, 5))

plt.hist(
    df_clean["Marks"],
    bins=6,
    edgecolor="black"
)

plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.title("Marks Distribution")

plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.close()

print("Saved marks_histogram.png")


# 6. Boxplot of Marks

plt.figure(figsize=(6, 5))

plt.boxplot(df_clean["Marks"])

plt.ylabel("Marks")
plt.title("Marks Boxplot")

plt.tight_layout()
plt.savefig("marks_boxplot.png")
plt.close()

print("Saved marks_boxplot.png")


# 7. Scatter plot: Study Hours vs Marks

plt.figure(figsize=(7, 5))

plt.scatter(
    df_clean["StudyHours"],
    df_clean["Marks"]
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")

plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")
plt.close()

print("Saved studyhours_vs_marks.png")


# 8. Correlation matrix

numeric_columns = [
    "StudyHours",
    "Attendance(%)",
    "Marks"
]

corr_matrix = df_clean[numeric_columns].corr()

print("\nCorrelation matrix:")
print(corr_matrix)


# 9. Gender-wise analysis

gender_summary = df_clean.groupby("Gender")[
    ["Marks", "Attendance(%)"]
].mean()

print("\nAverage Marks and Attendance by Gender:")
print(gender_summary)


# 10. Final dataset

print("\nCleaned Dataset:")
print(df_clean)