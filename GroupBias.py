import pandas as pd

# === Load data ===
cf_df = pd.read_csv("counterfactual_analysis.csv")

# === Map occupation codes to names ===
occupation_map = {
    '0': 'other', '1': 'academic/educator', '2': 'artist', '3': 'clerical/admin',
    '4': 'college/grad student', '5': 'customer service', '6': 'doctor/health care',
    '7': 'executive/managerial', '8': 'farmer', '9': 'homemaker', '10': 'K-12 student',
    '11': 'lawyer', '12': 'programmer', '13': 'retired', '14': 'sales/marketing',
    '15': 'scientist', '16': 'self-employed', '17': 'technician/engineer',
    '18': 'tradesman/craftsman', '19': 'unemployed', '20': 'writer'
}

cf_df['occupation'] = cf_df['occupation'].astype(str)
cf_df['occupation_name'] = cf_df['occupation'].map(occupation_map).fillna('unknown')

# === Gender Bias Summary ===
gender_summary = cf_df.groupby('gender')['changed'].mean() * 100
gender_summary_df = gender_summary.reset_index().rename(columns={'changed': 'Bias Percentage'})

# === Age Group Bias Summary ===
age_bins = [0, 18, 25, 35, 45, 55, 100]
age_labels = ['<18', '18–24', '25–34', '35–44', '45–54', '55+']
cf_df['age_group'] = pd.cut(cf_df['age'], bins=age_bins, labels=age_labels)
age_summary = cf_df.groupby('age_group')['changed'].mean() * 100
age_summary_df = age_summary.reset_index().rename(columns={'changed': 'Bias Percentage'})

# === Occupation Bias Summary (Top 10 only) ===
occupation_summary = (
    cf_df.groupby('occupation_name')['changed']
    .mean()
    .sort_values(ascending=False) * 100
)
occupation_summary_df = occupation_summary.reset_index().rename(columns={'changed': 'Bias Percentage'})
occupation_summary_df = occupation_summary_df[occupation_summary_df['Bias Percentage'] > 0]

# ✅ Limit to Top 10 most biased occupations
occupation_summary_df = occupation_summary_df.head(10)

# === Save summaries ===
gender_summary_df.to_csv("gender_bias_summary.csv", index=False)
age_summary_df.to_csv("age_bias_summary.csv", index=False)
occupation_summary_df.to_csv("occupation_bias_summary.csv", index=False)

# === Display summaries ===
print("\nGroup-Level Gender Bias Summary:\n", gender_summary_df)
print("\nGroup-Level Age Group Bias Summary:\n", age_summary_df)
print("\nTop 10 Group-Level Occupation Bias Summary:\n", occupation_summary_df)
print("\n✅ Group bias summaries saved with Top 10 occupations.")
