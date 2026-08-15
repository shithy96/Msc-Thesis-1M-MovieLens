import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load counterfactual results ===
cf_df = pd.read_csv("counterfactual_analysis.csv")

# === Calculate individual bias reduction ===
original_bias = cf_df['changed'].mean() * 100
post_bias = ((cf_df['fair_pred'] != cf_df['original_pred']) &
             (cf_df['fair_pred'] != cf_df['ground_truth'])).mean() * 100
bias_reduction = original_bias - post_bias

# === Enhanced Visualization ===
plt.figure(figsize=(8, 6))

# Create bar plot with better spacing
bars = plt.bar(
    ['Original Bias', 'Post-Correction Bias'],
    [original_bias, post_bias],
    color=['#e74c3c', '#2ecc71'],
    width=0.6  # Wider bars for better visibility
)

# Add value labels with improved positioning
for i, bar in enumerate(bars):
    height = bar.get_height()
    offset = max(original_bias, post_bias) * 0.05  # Dynamic offset
    plt.text(bar.get_x() + bar.get_width()/2.,
             height + offset,
             f'{height:.1f}%',
             ha='center',
             va='bottom',
             fontsize=16)

# Add horizontal line at zero for reference
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Formatting improvements
plt.title('Bias Reduction Through Counterfactual Correction', fontsize=17, pad=20)
plt.ylabel('Bias Percentage (%)', fontsize=15, labelpad=10)
plt.ylim(0, max(original_bias, post_bias) * 1.5)  # More headroom
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Remove spines for cleaner look
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(fontsize=14)
plt.tight_layout()
plt.savefig("individual_bias_reduction.png", dpi=300, bbox_inches='tight')
plt.close()

# === Print Results ===
print(f"""
✅ Fairness Report:
- Original Bias Percentage: {original_bias:.1f}%
- Post-Correction Bias Percentage: {post_bias:.1f}%
- Bias Reduction Achieved: {bias_reduction:.1f}%

📊 Chart saved: individual_bias_reduction.png
""")