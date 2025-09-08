import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
df = pd.read_csv('C:\\Users\\91740\\Desktop\\tasks\\titanic.csv')

# Step 2: Check column order
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Suppose the order is as in your dataset:
# 0: tPassengerId
# 1: tSurvived
# 2: tPclass
# 3: tName
# 4: tSex
# 5: tAge
# 6: tSibSp
# 7: tParch
# 8: tTicket
# 9: tFare
# 10: tCabin
# 11: tEmbarked

# Step 3: Data Cleaning by Index
# Fill missing tAge values (index 5) with the median
df.iloc[:, 5] = df.iloc[:, 5].fillna(df.iloc[:, 5].median())

# Fill missing tEmbarked values (index 11) with the mode
df.iloc[:, 11] = df.iloc[:, 11].fillna(df.iloc[:, 11].mode())

# Drop columns tCabin (10), tTicket (8), tName (3)
df.drop(df.columns[[10, 8, 3]], axis=1, inplace=True)

# Drop remaining rows with missing data
df.dropna(inplace=True)

#Step 4: Attractive Plot Styling
sns.set_theme(style="darkgrid", palette="pastel", font_scale=1.2)

# Survival count by Gender (tSurvived: 1, tSex: 4)
plt.figure(figsize=(10,6))
sns.countplot(x=df.columns[1], hue=df.columns[4], data=df, palette="Set2", edgecolor="black")
plt.title('Survival Count by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Survived', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title="Gender",loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()

# Survival count by Passenger Class (tSurvived: 1, tPclass: 2)
plt.figure(figsize=(10,6))
sns.countplot(x=df.columns[1], hue=df.columns[2], data=df, palette="Set1", edgecolor="black")
plt.title('Survival Count by Passenger Class', fontsize=16, fontweight='bold')
plt.xlabel('Survived', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title="Pclass")
plt.tight_layout()
plt.show()

# Distribution of Age (tAge: 5)
plt.figure(figsize=(10,6))
sns.histplot(df.iloc[:, 5], bins=30, kde=True, color="skyblue", edgecolor='black')
plt.title('Age Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.show()

# Correlation Heatmap (numeric columns only)
plt.figure(figsize=(8,6))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu", linewidths=1, linecolor='white', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Step 5: Save Cleaned Data (Optional)
df.to_csv('titanic_cleaned.csv', index=False)
