import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the counterfactual analysis file
df = pd.read_csv("counterfactual_analysis.csv")

# Compute accuracy scores
original_accuracy = accuracy_score(df['ground_truth'], df['original_pred']) * 100
fair_accuracy = accuracy_score(df['ground_truth'], df['fair_pred']) * 100

# Print values
print(f"Original Accuracy: {original_accuracy:.2f}%")
print(f"Fair Prediction Accuracy: {fair_accuracy:.2f}%")

# Bar plot to compare accuracies
plt.figure(figsize=(6.5, 5))
bars = plt.bar(['Original', 'Fair'], [original_accuracy, fair_accuracy],
               color=['#3498db', '#2ecc71'])

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%',
             ha='center', va='bottom', fontsize=12)

plt.title('Accuracy Comparison: Original vs Fair Predictions', fontsize=13)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save figure
plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("📊 Accuracy comparison chart saved as 'accuracy_comparison.png'")
