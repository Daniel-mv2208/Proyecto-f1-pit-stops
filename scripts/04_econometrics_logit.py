import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting econometric analysis (Logit model with dedicated directory)...")

# 1. Ensure the 'econometrics' subfolder exists within 'outputs'
output_dir = os.path.join('outputs', 'econometrics')
os.makedirs(output_dir, exist_ok=True)

# 2. Load the master dataset
df = pd.read_csv('data/f1_master_dataset.csv')

# 3. Logit model specification
formula = 'target_top5 ~ delta_pit_stop + total_stops'
logit_model = smf.logit(formula=formula, data=df).fit()

# 4. Export standard statistical summary
summary = logit_model.summary()
report_path = os.path.join(output_dir, 'logit_econometrics_report.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=== ECONOMETRIC REPORT: LOGISTIC REGRESSION (F1) ===\n")
    f.write("Based on the qualitative response model framework (D. Gujarati)\n\n")
    f.write(str(summary))
    
    # Calculate and append odds ratios
    f.write("\n\n=== ODDS RATIOS ===\n")
    odds_ratios = np.exp(logit_model.params)
    f.write(str(odds_ratios))

print(f"Statistical report saved to: {report_path}")

# 5. Generate econometric visualizations
sns.set_theme(style="whitegrid")

# Probability sigmoid curve
plt.figure(figsize=(10, 6))
median_stops = df['total_stops'].median()
delta_range = np.linspace(df['delta_pit_stop'].min(), df['delta_pit_stop'].max(), 300)
pred_data = pd.DataFrame({'delta_pit_stop': delta_range, 'total_stops': median_stops})

probabilities = logit_model.predict(pred_data)

plt.plot(delta_range, probabilities, color='crimson', linewidth=3, label=f'Logit Curve (Stops = {median_stops})')
plt.scatter(df['delta_pit_stop'], df['target_top5'], alpha=0.1, color='black', label='Observed Data')
plt.axvline(x=0, color='gray', linestyle='--', label='Race Baseline (Delta = 0)')
plt.title('Top 5 Finish Probability by Pit Stop Delta', fontsize=14, fontweight='bold')
plt.xlabel('Pit Stop Delta (Seconds Relative to Race Average)', fontsize=12)
plt.ylabel('Estimated Probability P(Top 5 = 1)', fontsize=12)
plt.legend()
plt.tight_layout()

plot_1_path = os.path.join(output_dir, 'logit_probability_curve.png')
plt.savefig(plot_1_path, dpi=300)
plt.close()

# Forest plot of Odds Ratios
plt.figure(figsize=(8, 4))
conf = logit_model.conf_int()
conf['Odds Ratio'] = odds_ratios
conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
conf = conf.drop('Intercept')

lower_errors = conf['Odds Ratio'] - np.exp(conf['2.5%'])
upper_errors = np.exp(conf['97.5%']) - conf['Odds Ratio']

plt.errorbar(
    x=conf['Odds Ratio'], 
    y=conf.index, 
    xerr=[lower_errors, upper_errors], 
    fmt='o', 
    color='navy', 
    markersize=10, 
    capsize=5, 
    linewidth=2
)
plt.axvline(x=1, color='red', linestyle='--', label='No Effect (OR = 1)')
plt.title('Feature Impact (Odds Ratios)', fontsize=14, fontweight='bold')
plt.xlabel('Probability Multiplier (Odds Ratio)', fontsize=12)
plt.legend()
plt.tight_layout()

plot_2_path = os.path.join(output_dir, 'odds_ratios_impact.png')
plt.savefig(plot_2_path, dpi=300)
plt.close()

print(f"Econometric plots saved successfully to: {output_dir}")