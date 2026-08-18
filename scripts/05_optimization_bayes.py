import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting business optimization analysis (Bayes' Theorem)...")

# 1. Ensure the 'optimization' subfolder exists within 'outputs'
output_dir = os.path.join('outputs', 'optimization')
os.makedirs(output_dir, exist_ok=True)

# 2. Load master dataset
df = pd.read_csv('data/f1_master_dataset.csv')

# 3. Define strategic events
total_team_races = len(df)
is_top5 = df['target_top5'] == 1
is_efficient = df['delta_pit_stop'] < 0.0

# 4. Calculate baseline probabilities
p_a = df[is_top5].shape[0] / total_team_races
p_b = df[is_efficient].shape[0] / total_team_races
p_b_given_a = df[is_top5 & is_efficient].shape[0] / df[is_top5].shape[0]

# 5. Apply Bayes' Theorem
p_a_given_b = (p_b_given_a * p_a) / p_b
strategic_lift = p_a_given_b - p_a

# 6. Generate and export textual report
report_path = os.path.join(output_dir, 'bayes_optimization_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=== BUSINESS OPTIMIZATION REPORT: BAYES' THEOREM ===\n")
    f.write("Decision variable: Maintaining a negative pit stop delta (< 0.0s)\n")
    f.write("-" * 70 + "\n\n")
    f.write("1. BASE PROBABILITIES:\n")
    f.write(f"   - P(A)   | Baseline probability of finishing Top 5:            {p_a:.1%}\n")
    f.write(f"   - P(B)   | Probability of achieving a fast pit stop:           {p_b:.1%}\n")
    f.write(f"   - P(B|A) | Share of Top 5 teams that achieve fast pit stops:   {p_b_given_a:.1%}\n\n")
    f.write("2. BAYES' THEOREM APPLICATION P(A|B):\n")
    f.write(f"   - Probability of Top 5 finish GIVEN negative delta:            {p_a_given_b:.1%}\n\n")
    f.write("3. STRATEGIC TAKEAWAY:\n")
    f.write(f"   The absolute strategic lift is {strategic_lift:.1%}.\n")

# 7. Visualization 1: Strategic impact comparison
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

impact_scenarios = ['Baseline Probability\n(P(A) Overall)', 'Optimized Probability\n(P(A|B) Bayesian)']
impact_probs = [p_a * 100, p_a_given_b * 100]
impact_colors = ['#95a5a6', '#27ae60']

impact_bars = plt.bar(impact_scenarios, impact_probs, color=impact_colors, width=0.5)

for bar in impact_bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0, 
        height + 1.5,
        f'{height:.1f}%', 
        ha='center', 
        va='bottom', 
        fontweight='bold', 
        fontsize=12
    )

plt.annotate(
    f'Lift:\n+{strategic_lift * 100:.1f}%', 
    xy=(0.5, p_a * 100), 
    xytext=(0.5, (p_a * 100) + 10),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    ha='center', 
    va='center', 
    fontweight='bold', 
    color='black',
    bbox=dict(boxstyle="round,pad=0.3", fc="#f1c40f", ec="black", lw=1)
)

plt.title('Strategic Impact of Pit Stop Efficiency', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Probability of Finishing in Top 5 (%)', fontsize=12)
plt.ylim(0, 100)
plt.tight_layout()
plot_1_path = os.path.join(output_dir, 'bayesian_impact_comparison.png')
plt.savefig(plot_1_path, dpi=300)
plt.close()

# 8. Visualization 2: Fundamental probability breakdown
plt.figure(figsize=(9, 6))

base_labels = [
    'P(A)\nProbability of\nTop 5 Finish', 
    'P(B)\nProbability of\nNegative Delta', 
    'P(B|A)\nNegative Delta given\nTop 5 Finish'
]
base_values = [p_a * 100, p_b * 100, p_b_given_a * 100]
base_colors = ['#3498db', '#34495e', '#8e44ad']

base_bars = plt.bar(base_labels, base_values, color=base_colors, width=0.6)

for bar in base_bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0, 
        height + 1.5,
        f'{height:.1f}%', 
        ha='center', 
        va='bottom', 
        fontweight='bold', 
        fontsize=12
    )

plt.title('Breakdown of Fundamental Probabilities (Bayes)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Probability (%)', fontsize=12)
plt.ylim(0, 100)
plt.tight_layout()
plot_2_path = os.path.join(output_dir, 'base_probabilities_breakdown.png')
plt.savefig(plot_2_path, dpi=300)
plt.close()

print("Bayesian analysis complete.")
print(f"Report saved to: {report_path}")
print(f"Impact comparison plot saved to: {plot_1_path}")
print(f"Base probabilities breakdown plot saved to: {plot_2_path}")