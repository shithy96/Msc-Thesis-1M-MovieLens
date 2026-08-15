import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load bias summaries ===
cf_df = pd.read_csv("counterfactual_analysis.csv")
gender_summary = pd.read_csv("gender_bias_summary.csv")
age_summary = pd.read_csv("age_bias_summary.csv")
occupation_summary = pd.read_csv("occupation_bias_summary.csv")

# === 1. Individual Bias (Biased vs Unbiased) ===
changed = cf_df['changed'].sum()
unchanged = len(cf_df) - changed
total = len(cf_df)

plt.figure(figsize=(6, 5))
bars = plt.bar(['Biased', 'Unbiased'], [changed, unchanged],
               color=['#e74c3c', '#3498db'])

for bar in bars:
    height = bar.get_height()
    percentage = height / total * 100
    plt.text(bar.get_x() + bar.get_width()/2, height + 5,
             f'{percentage:.1f}%',
             ha='center', va='bottom', fontsize=12)

plt.title('Individual-Level Bias (After Gender Flip)', fontsize=14)
plt.ylabel('User Count', fontsize=12)
plt.ylim(0, max(changed, unchanged) * 1.2)
plt.tight_layout()
plt.savefig("individual_bias_bar.png", dpi=300)
plt.close()

# === 2. Group-Level Gender Bias ===
plt.figure(figsize=(7, 5))
bars = plt.bar(
    gender_summary['gender'],
    gender_summary['Bias Percentage'],
    color=['#f39c12', '#8e44ad']
)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10)

plt.title('Group-Level Bias by Gender', fontsize=14)
plt.xlabel('Gender')
plt.ylabel('Bias Percentage (%)')
plt.ylim(0, max(gender_summary['Bias Percentage'].max() * 1.2, 20))
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("group_gender_bias_bar.png", dpi=300)
plt.close()

# === 3. Group-Level Age Bias ===
plt.figure(figsize=(10, 6))
bars = plt.bar(
    age_summary['age_group'],
    age_summary['Bias Percentage'],
    color='#2ecc71'
)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10)

plt.title('Group-Level Bias by Age Group', fontsize=14)
plt.xlabel('Age Group')
plt.ylabel('Bias Percentage (%)')
plt.ylim(0, max(age_summary['Bias Percentage'].max() * 1.2, 25))
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("group_age_bias_bar.png", dpi=300)
plt.close()

# === 4. Group-Level Occupation Bias (Top 10) ===
plt.figure(figsize=(12, 8))
bars = plt.barh(
    occupation_summary['occupation_name'],
    occupation_summary['Bias Percentage'],
    color='#9b59b6'
)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2,
             f'{width:.1f}%',
             ha='left', va='center', fontsize=12)

plt.title('Top 10 Group-Level Bias by Occupation', fontsize=14)
plt.xlabel('Bias Percentage (%)')
plt.xlim(0, max(occupation_summary['Bias Percentage'].max() * 1.15, 80))
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("group_occupation_bias_bar.png", dpi=300)
plt.close()

print("✅ All bias charts saved successfully.")
