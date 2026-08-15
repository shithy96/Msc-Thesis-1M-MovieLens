import pandas as pd
import numpy as np

print("=" * 80)
print("FAIRNESS METRICS CALCULATION")
print("Calculating POR and DPD for ALL groups")
print("=" * 80)

# === Load the counterfactual analysis data ===
cf_df = pd.read_csv("counterfactual_analysis.csv")

# === Create age groups (same as in GroupBias.py) ===
age_bins = [0, 18, 25, 35, 45, 55, 100]
age_labels = ['<18', '18–24', '25–34', '35–44', '45–54', '55+']
cf_df['age_group'] = pd.cut(cf_df['age'], bins=age_bins, labels=age_labels)

# === Map occupation codes to names ===
occupation_map = {
    '0': 'other', '1': 'academic/educator', '2': 'artist', '3': 'clerical/admin',
    '4': 'college/grad student', '5': 'customer service', '6': 'doctor/health care',
    '7': 'executive/managerial', '8': 'farmer', '9': 'homemaker', '10': 'K-12 student',
    '11': 'lawyer', '12': 'programmer', '13': 'retired', '14': 'sales/marketing',
    '15': 'scientist', '16': 'self-employed', '17': 'technician/engineer',
    '18': 'tradesman/craftsman', '19': 'unemployed', '20': 'writer'
}
cf_df['occupation_name'] = cf_df['occupation'].astype(str).map(occupation_map).fillna('unknown')


# === Function to calculate Demographic Parity Difference ===
def calculate_dpd(df, group_col, pred_col):
    """Calculate Demographic Parity Difference for a given group and prediction column."""
    group_stats = df.groupby(group_col)[pred_col].mean()
    dpd = group_stats.max() - group_stats.min()
    return dpd, group_stats


# === Function to calculate Positive Outcome Rate per group ===
def calculate_por(df, group_col, pred_col):
    """Calculate Positive Outcome Rate for each group."""
    por = df.groupby(group_col)[pred_col].mean() * 100  # Convert to percentage
    return por


# === Get the SAME top 10 occupations from group bias file ===
def get_same_top_10_occupations():
    """Load the SAME top 10 occupations from occupation_bias_summary.csv."""
    try:
        # This is the file created by GroupBias.py
        top10_df = pd.read_csv("occupation_bias_summary.csv")
        top10_occupations = top10_df['occupation_name'].tolist()
        print(f"✅ Loaded {len(top10_occupations)} occupations from occupation_bias_summary.csv")
        return top10_occupations
    except FileNotFoundError:
        print("⚠️  occupation_bias_summary.csv not found. Calculating top 10 by bias...")
        # Fallback: Calculate top 10 by bias
        occupation_bias = cf_df.groupby('occupation_name')['changed'].mean() * 100
        top10_occupations = occupation_bias.nlargest(10).index.tolist()
        return top10_occupations


# === Get the SAME top 10 occupations ===
same_top10_occupations = get_same_top_10_occupations()
print("\nSAME Top 10 Occupations (for visualization consistency):")
for i, occ in enumerate(same_top10_occupations, 1):
    print(f"  {i:2d}. {occ}")

# === Calculate metrics for ALL groups ===
print("\n" + "=" * 80)
print("CALCULATING METRICS FOR ALL GROUPS")
print("=" * 80)

metrics_results = []

for attribute in ['gender', 'age_group', 'occupation_name']:
    # Calculate DPD for ALL groups
    dpd_original, _ = calculate_dpd(cf_df, attribute, 'original_pred')
    dpd_fair, _ = calculate_dpd(cf_df, attribute, 'fair_pred')

    # Calculate POR for ALL groups
    por_original = calculate_por(cf_df, attribute, 'original_pred')
    por_fair = calculate_por(cf_df, attribute, 'fair_pred')

    # Store results
    attribute_name = attribute
    if attribute == 'occupation_name':
        attribute_name = 'occupation_name_all'  # Indicate these are ALL occupations

    metrics_results.append({
        'Attribute': attribute_name,
        'DPD_Original': dpd_original,
        'DPD_Fair': dpd_fair,
        'DPD_Reduction': dpd_original - dpd_fair,
        'POR_Original_Avg': por_original.mean(),
        'POR_Fair_Avg': por_fair.mean(),
        'N_Groups': len(por_original)
    })

    print(f"\n{attribute.upper()} (ALL {len(por_original)} groups):")
    print(f"  DPD: {dpd_original:.4f} → {dpd_fair:.4f} (Reduction: {dpd_original - dpd_fair:.4f})")
    print(f"  POR Range: {por_original.min():.1f}%-{por_original.max():.1f}%")

# === Create summary DataFrame ===
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv("fairness_metrics_summary_all.csv", index=False)

print("\n" + "=" * 80)
print("DETAILED POSITIVE OUTCOME RATES PER GROUP")
print("=" * 80)

# === Calculate and display detailed POR for each group ===

# 1. Gender detailed POR
print("\n" + "-" * 60)
print("GENDER - POSITIVE OUTCOME RATE PER GROUP")
print("-" * 60)
por_original_gender = calculate_por(cf_df, 'gender', 'original_pred')
por_fair_gender = calculate_por(cf_df, 'gender', 'fair_pred')

for gender in por_original_gender.index:
    orig = por_original_gender[gender]
    fair = por_fair_gender[gender]
    change = fair - orig
    print(f"  {gender:6s}: Original = {orig:6.2f}% | Fair = {fair:6.2f}% | Change = {change:+6.2f}%")

# 2. Age group detailed POR
print("\n" + "-" * 60)
print("AGE GROUPS - POSITIVE OUTCOME RATE PER GROUP")
print("-" * 60)
por_original_age = calculate_por(cf_df, 'age_group', 'original_pred')
por_fair_age = calculate_por(cf_df, 'age_group', 'fair_pred')

# Sort age groups in logical order
age_order = ['<18', '18–24', '25–34', '35–44', '45–54', '55+']
for age_group in age_order:
    if age_group in por_original_age.index:
        orig = por_original_age[age_group]
        fair = por_fair_age[age_group]
        change = fair - orig
        print(f"  {age_group:6s}: Original = {orig:6.2f}% | Fair = {fair:6.2f}% | Change = {change:+6.2f}%")

# 3. All occupations detailed POR
print("\n" + "-" * 60)
print("ALL OCCUPATIONS - POSITIVE OUTCOME RATE PER GROUP")
print("-" * 60)
print("(Sorted by Original POR)")
por_original_occ_all = calculate_por(cf_df, 'occupation_name', 'original_pred')
por_fair_occ_all = calculate_por(cf_df, 'occupation_name', 'fair_pred')

# Create DataFrame for sorting
occ_df = pd.DataFrame({
    'occupation': por_original_occ_all.index,
    'POR_Original': por_original_occ_all.values,
    'POR_Fair': por_fair_occ_all.values
})
occ_df['Change'] = occ_df['POR_Fair'] - occ_df['POR_Original']
occ_df['Is_Top10'] = occ_df['occupation'].isin(same_top10_occupations)

# Sort by original POR
occ_df_sorted = occ_df.sort_values('POR_Original', ascending=False)

# Display with indication of top 10
for idx, row in occ_df_sorted.iterrows():
    top10_marker = " [TOP 10]" if row['Is_Top10'] else ""
    print(f"  {row['occupation']:25s}: Original = {row['POR_Original']:6.2f}% | "
          f"Fair = {row['POR_Fair']:6.2f}% | Change = {row['Change']:+6.2f}%{top10_marker}")

print("\n" + "=" * 80)
print("FAIRNESS METRICS SUMMARY (ALL GROUPS)")
print("=" * 80)

for idx, row in metrics_df.iterrows():
    print(f"\n{row['Attribute'].replace('_all', '').upper()} Analysis:")
    print(f"  Demographic Parity Difference:")
    print(f"    Original: {row['DPD_Original']:.4f}")
    print(f"    After Fairness: {row['DPD_Fair']:.4f}")
    if row['DPD_Original'] > 0:
        improvement = (row['DPD_Reduction'] / row['DPD_Original']) * 100
        print(f"    Reduction: {row['DPD_Reduction']:.4f} ({improvement:.1f}%)")
    print(f"  Positive Outcome Rate (Average):")
    print(f"    Original: {row['POR_Original_Avg']:.2f}%")
    print(f"    After Fairness: {row['POR_Fair_Avg']:.2f}%")

print("\n" + "=" * 80)
print("SAVING DETAILED POR FOR ALL GROUPS")
print("=" * 80)

# 1. Save Gender detailed POR
gender_df = pd.DataFrame({
    'gender': por_original_gender.index,
    'POR_Original_%': por_original_gender.values.round(2),
    'POR_Fair_%': por_fair_gender.values.round(2),
    'Change_%': (por_fair_gender.values - por_original_gender.values).round(2)
})
gender_df.to_csv("por_detailed_gender_all.csv", index=False)
print("✅ Saved: por_detailed_gender_all.csv")

# 2. Save Age Groups detailed POR
age_df = pd.DataFrame({
    'age_group': por_original_age.index,
    'POR_Original_%': por_original_age.values.round(2),
    'POR_Fair_%': por_fair_age.values.round(2),
    'Change_%': (por_fair_age.values - por_original_age.values).round(2)
})
age_df.to_csv("por_detailed_age_all.csv", index=False)
print("✅ Saved: por_detailed_age_all.csv")

# 3. Save ALL Occupations detailed POR
occ_all_df = pd.DataFrame({
    'occupation_name': por_original_occ_all.index,
    'POR_Original_%': por_original_occ_all.values.round(2),
    'POR_Fair_%': por_fair_occ_all.values.round(2),
    'Change_%': (por_fair_occ_all.values - por_original_occ_all.values).round(2),
    'Is_Top10': por_original_occ_all.index.isin(same_top10_occupations)
}).sort_values('POR_Original_%', ascending=False)

occ_all_df.to_csv("por_detailed_occupation_all.csv", index=False)
print("✅ Saved: por_detailed_occupation_all.csv (ALL occupations)")

# 4. Save SAME Top 10 Occupations (filtered from all)
occ_top10_df = occ_all_df[occ_all_df['occupation_name'].isin(same_top10_occupations)].copy()
occ_top10_df = occ_top10_df.sort_values('POR_Original_%', ascending=False)
occ_top10_df.to_csv("por_detailed_occupation_same_top10.csv", index=False)
print("✅ Saved: por_detailed_occupation_same_top10.csv (SAME Top 10 from group bias)")

# === Final statistics ===
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)

print(f"\nTotal Samples: {len(cf_df):,}")
print(f"Gender Groups: {len(gender_df)}")
print(f"Age Groups: {len(age_df)}")
print(f"Occupation Groups: {len(occ_all_df)} (ALL)")
print(f"Top 10 Occupations (for visualization): {len(occ_top10_df)}")

# Calculate coverage
top10_samples = cf_df[cf_df['occupation_name'].isin(same_top10_occupations)].shape[0]
coverage = (top10_samples / len(cf_df)) * 100
print(f"\nTop 10 Coverage: {top10_samples:,} samples ({coverage:.1f}% of total)")

# Summary of changes
print("\n" + "=" * 80)
print("SUMMARY OF CHANGES")
print("=" * 80)

# Gender changes
gender_changes = gender_df['Change_%'].values
print(f"\nGender Changes: {gender_changes[0]:+.2f}% (Female), {gender_changes[1]:+.2f}% (Male)")

# Age group changes
print(f"\nAge Group Changes (min to max):")
print(f"  Minimum change: {age_df['Change_%'].min():+.2f}% ({age_df.loc[age_df['Change_%'].idxmin(), 'age_group']})")
print(f"  Maximum change: {age_df['Change_%'].max():+.2f}% ({age_df.loc[age_df['Change_%'].idxmax(), 'age_group']})")

# Occupation changes
print(f"\nOccupation Changes (Top 5 improvements):")
top_improvements = occ_top10_df.nlargest(5, 'Change_%')
for idx, row in top_improvements.iterrows():
    print(
        f"  {row['occupation_name']:25s}: {row['Change_%']:+.2f}% ({row['POR_Original_%']:.1f}% → {row['POR_Fair_%']:.1f}%)")

print("\n" + "=" * 80)
print("CALCULATION COMPLETE!")
print("=" * 80)
print("\n📁 Generated Files:")
print("1. fairness_metrics_summary_all.csv - Main metrics for ALL groups")
print("2. por_detailed_gender_all.csv - POR for ALL gender groups")
print("3. por_detailed_age_all.csv - POR for ALL age groups")
print("4. por_detailed_occupation_all.csv - POR for ALL 21 occupations")
print("5. por_detailed_occupation_same_top10.csv - POR for SAME Top 10 occupations")
print("\n📊 Console Output Includes:")
print("• DPD values for each attribute")
print("• Detailed POR for each gender group")
print("• Detailed POR for each age group")
print("• Detailed POR for each occupation group (all 21)")
print("• Top 10 occupations highlighted")
print("• Summary statistics")
print("\n✅ Now run: python FairnessMetrics_Visualization.py")
print("=" * 80)
