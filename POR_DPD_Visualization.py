import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 80)
print("FAIRNESS METRICS VISUALIZATION")
print("Creating separate PNG files for each group")
print("=" * 80)

# === Load data ===
print("📊 Loading data...")
metrics_df = pd.read_csv("fairness_metrics_summary_all.csv")
gender_df = pd.read_csv("por_detailed_gender_all.csv")
age_df = pd.read_csv("por_detailed_age_all.csv")
occupation_df = pd.read_csv("por_detailed_occupation_same_top10.csv")

print(f"✅ Loaded data:")
print(f"   - Gender groups: {len(gender_df)}")
print(f"   - Age groups: {len(age_df)}")
print(f"   - Occupations: {len(occupation_df)} (SAME Top 10 from group bias)")

# === 1. GENDER VISUALIZATION ===
print("\n1. Creating Gender visualization...")
plt.figure(figsize=(10, 7))

x = np.arange(len(gender_df))
width = 0.35

bars_original = plt.bar(x - width / 2, gender_df['POR_Original_%'], width,
                        label='Original', color='#FF6B6B', alpha=0.9,
                        edgecolor='black', linewidth=1.5)
bars_fair = plt.bar(x + width / 2, gender_df['POR_Fair_%'], width,
                    label='Fair', color='#4ECDC4', alpha=0.9,
                    edgecolor='black', linewidth=1.5)

plt.xlabel('Gender', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Positive Outcome Rate (%)', fontsize=14, fontweight='bold', labelpad=15)
plt.title('Positive Outcome Rate by Gender\nBefore vs After Fairness Correction',
          fontsize=16, fontweight='bold', pad=20)

plt.xticks(x, gender_df['gender'], fontsize=12)
plt.yticks(fontsize=11)
plt.legend(fontsize=12, loc='best', frameon=True)

# Add value labels
for bar in bars_original:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

for bar in bars_fair:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

# Add DPD info
gender_metrics = metrics_df[metrics_df['Attribute'] == 'gender'].iloc[0]
dpd_text = (f"DPD: {gender_metrics['DPD_Original']:.4f} → {gender_metrics['DPD_Fair']:.4f} "
            f"({gender_metrics['DPD_Reduction']:.4f} reduction)")
plt.figtext(0.5, 0.01, dpd_text, ha='center', fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("Gender_POR_Comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: Gender_POR_Comparison.png")

# === 2. AGE GROUP VISUALIZATION ===
print("\n2. Creating Age Group visualization...")
plt.figure(figsize=(12, 8))

x = np.arange(len(age_df))
width = 0.35

bars_original = plt.bar(x - width  / 2, age_df['POR_Original_%'], width,
                        label='Original', color='#3498DB', alpha=0.9,
                        edgecolor='black', linewidth=1.5)
bars_fair = plt.bar(x + width / 2, age_df['POR_Fair_%'], width,
                    label='Fair', color='#9B59B6', alpha=0.9,
                    edgecolor='black', linewidth=1.5)

plt.xlabel('Age Group', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Positive Outcome Rate (%)', fontsize=14, fontweight='bold', labelpad=15)
plt.title('Positive Outcome Rate by Age Group\nBefore vs After Fairness Correction',
          fontsize=16, fontweight='bold', pad=20)

plt.xticks(x, age_df['age_group'], fontsize=11, rotation=45, ha='right')
plt.yticks(fontsize=11)
plt.legend(fontsize=12, loc='best', frameon=True)

# Add value labels
for bar in bars_original:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

for bar in bars_fair:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

# Add DPD info
age_metrics = metrics_df[metrics_df['Attribute'] == 'age_group'].iloc[0]
dpd_text = (f"DPD: {age_metrics['DPD_Original']:.4f} → {age_metrics['DPD_Fair']:.4f} "
            f"({age_metrics['DPD_Reduction']:.4f} reduction)")
plt.figtext(0.5, 0.01, dpd_text, ha='center', fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("AgeGroup_POR_Comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: AgeGroup_POR_Comparison.png")

# === 3. OCCUPATION VISUALIZATION (SAME Top 10) ===
print("\n3. Creating Occupation visualization (SAME Top 10)...")

# Sort for better visualization (highest POR at top)
occupation_df = occupation_df.sort_values('POR_Original_%', ascending=True)

plt.figure(figsize=(14, 10))

y = np.arange(len(occupation_df))
height = 0.35

bars_original = plt.barh(y - height / 2, occupation_df['POR_Original_%'], height,
                         label='Original', color='#E67E22', alpha=0.9,
                         edgecolor='black', linewidth=1.5)
bars_fair = plt.barh(y + height / 2, occupation_df['POR_Fair_%'], height,
                     label='Fair', color='#27AE60', alpha=0.9,
                     edgecolor='black', linewidth=1.5)

plt.xlabel('Positive Outcome Rate (%)', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Occupation', fontsize=14, fontweight='bold', labelpad=15)
plt.title(
    'Positive Outcome Rate by Occupation\nBefore vs After Fairness Correction',
    fontsize=16, fontweight='bold', pad=20)

# Format occupation names (truncate if too long)
occupation_names = []
for name in occupation_df['occupation_name']:
    if len(name) > 25:
        occupation_names.append(name[:22] + '...')
    else:
        occupation_names.append(name)

plt.yticks(y, occupation_names, fontsize=10)
plt.xticks(fontsize=11)
plt.legend(fontsize=12, loc='lower right', frameon=True)

# Add value labels
for bar in bars_original:
    width_val = bar.get_width()
    plt.text(width_val + 0.5, bar.get_y() + bar.get_height() / 2,
             f'{width_val:.1f}%', ha='left', va='center',
             fontsize=10, fontweight='bold')

for bar in bars_fair:
    width_val = bar.get_width()
    plt.text(width_val + 0.5, bar.get_y() + bar.get_height() / 2,
             f'{width_val:.1f}%', ha='left', va='center',
             fontsize=10, fontweight='bold')

# Add DPD info (using ALL occupations for accuracy)
occ_metrics = metrics_df[metrics_df['Attribute'] == 'occupation_name_all'].iloc[0]
dpd_text = (f"DPD (ALL 21 occupations): {occ_metrics['DPD_Original']:.4f} → {occ_metrics['DPD_Fair']:.4f} "
            f"({occ_metrics['DPD_Reduction']:.4f} reduction)")
plt.figtext(0.5, 0.01, dpd_text, ha='center', fontsize=11, fontweight='bold')

plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("Occupation_POR_Comparison_SameTop10.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: Occupation_POR_Comparison_SameTop10.png")

# === 4. DPD REDUCTION SUMMARY ===
print("\n4. Creating DPD Reduction summary...")
plt.figure(figsize=(10, 7))
# === 4. DPD REDUCTION SUMMARY ===
print("\n4. Creating DPD Reduction summary...")
plt.figure(figsize=(10, 7))

# Prepare data for display
display_names = ['Gender Group', 'Age Group', 'Occupation Group']
attributes = ['gender', 'age_group', 'occupation_name_all']

dpd_original = []
dpd_fair = []

for attr in attributes:
    row = metrics_df[metrics_df['Attribute'] == attr].iloc[0]
    dpd_original.append(row['DPD_Original'])
    dpd_fair.append(row['DPD_Fair'])

x = np.arange(len(display_names))
width = 0.35

bars_original = plt.bar(x - width / 2, dpd_original, width,
                        label='Original DPD', color='#E74C3C', alpha=0.9,
                        edgecolor='black', linewidth=1.5)
bars_fair = plt.bar(x + width / 2, dpd_fair, width,
                    label='Fair DPD', color='#2ECC71', alpha=0.9,
                    edgecolor='black', linewidth=1.5)

plt.xlabel('Attributes', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Demographic Parity Difference', fontsize=14, fontweight='bold', labelpad=15)
plt.title('Demographic Parity Difference Reduction\nAcross All Attributes',
          fontsize=16, fontweight='bold', pad=20)

plt.xticks(x, display_names, fontsize=12)
plt.yticks(fontsize=11)
plt.legend(fontsize=12, frameon=True)

# Add value labels ONLY (NO improvement percentage)
for i, (orig, fair) in enumerate(zip(dpd_original, dpd_fair)):
    # Original DPD value
    plt.text(i - width / 2, orig + 0.0005, f'{orig:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Fair DPD value
    plt.text(i + width / 2, fair + 0.0005, f'{fair:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("DPD_Reduction_Summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: DPD_Reduction_Summary.png")
# === 5. FAIRNESS IMPROVEMENT SUMMARY ===
print("\n5. Creating Fairness Improvement summary...")
plt.figure(figsize=(8, 6))

improvements = []
labels = []

for attr in attributes:
    row = metrics_df[metrics_df['Attribute'] == attr].iloc[0]
    if row['DPD_Original'] > 0:
        improvement = (row['DPD_Reduction'] / row['DPD_Original']) * 100
        improvements.append(improvement)
        label = attr.replace('_all', '').replace('_', ' ').title()
        if attr == 'occupation_name_all':
            label = 'Occupation Group'
        labels.append(label)

colors = ['#FF6B6B', '#3498DB', '#27AE60'][:len(improvements)]

bars = plt.bar(labels, improvements, color=colors, alpha=0.9,
               edgecolor='black', linewidth=1.5)

plt.xlabel('Attribute', fontsize=12, fontweight='bold')
plt.ylabel('DPD Reduction (%)', fontsize=12, fontweight='bold')
plt.title('Fairness Improvement by Attribute', fontsize=14, fontweight='bold', pad=15)

# Add value labels
for i, (bar, improvement) in enumerate(zip(bars, improvements)):
    plt.text(bar.get_x() + bar.get_width() / 2, improvement + 0.5,
             f'{improvement:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("Fairness_Improvement_Summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: Fairness_Improvement_Summary.png")

# === PRINT SUMMARY ===
print("\n" + "=" * 80)
print("VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 80)
print("\n📁 SEPARATE PNG FILES CREATED:")
print("1. Gender_POR_Comparison.png - Gender POR before/after")
print("2. AgeGroup_POR_Comparison.png - Age group POR before/after")
print("3. Occupation_POR_Comparison_SameTop10.png - SAME Top 10 occupations (from group bias)")
print("4. DPD_Reduction_Summary.png - DPD reduction across all attributes")
print("5. Fairness_Improvement_Summary.png - Percentage improvement")
print("\n📊 KEY POINTS:")
print("• Calculations use ALL groups for accuracy")
print("• Occupation visualization shows SAME Top 10 as group bias analysis")
print("• DPD values calculated from ALL 21 occupations")
print("\n" + "=" * 80)
print("✅ All visualizations complete! Ready for thesis.")
print("=" * 80)
