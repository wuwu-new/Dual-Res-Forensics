#!/usr/bin/env python3
"""
Generate evaluation visualizations for DRF v2 using summary metrics.
Creates comparison plots and performance analysis charts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTS_DIR = Path("logs/drf_v2_remote_vitl14")
OUTPUT_DIR = RESULTS_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load summary results
with open(RESULTS_DIR / "test_result_model.json") as f:
    model_result = json.load(f)

with open(RESULTS_DIR / "test_result_tta.json") as f:
    tta_result = json.load(f)

print("✓ Loaded model and TTA results")

# Extract metrics
def extract_metrics(result):
    """Extract per-dataset and average metrics."""
    datasets = result["per_dataset"]
    return {
        "cdfv2": {
            "AUC": datasets["cdfv2"]["auc"],
            "AP": datasets["cdfv2"]["ap"],
            "EER": datasets["cdfv2"]["eer"] * 100,
            "ACC": datasets["cdfv2"]["acc"],
            "F1": datasets["cdfv2"]["best_f1"],
        },
        "dfdc": {
            "AUC": datasets["dfdc"]["auc"],
            "AP": datasets["dfdc"]["ap"],
            "EER": datasets["dfdc"]["eer"] * 100,
            "ACC": datasets["dfdc"]["acc"],
            "F1": datasets["dfdc"]["best_f1"],
        },
        "avg_auc": result["avg_auc"],
    }

model_metrics = extract_metrics(model_result)
tta_metrics = extract_metrics(tta_result)

print(f"✓ Model avg AUC: {model_metrics['avg_auc']:.4f}")
print(f"✓ TTA avg AUC: {tta_metrics['avg_auc']:.4f}")

# ============================================================================
# 1. MAIN METRICS COMPARISON
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('DRF v2 remote_vitl14: Model vs TTA Performance Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

metrics_list = ["AUC", "AP", "ACC", "F1"]
datasets = ["cdfv2", "dfdc"]

for idx, metric in enumerate(metrics_list):
    ax = axes[idx // 2, idx % 2]
    
    # Prepare data
    x = np.arange(len(datasets))
    width = 0.35
    
    model_vals = [model_metrics[ds][metric] for ds in datasets]
    tta_vals = [tta_metrics[ds][metric] for ds in datasets]
    
    # Create bars
    bars1 = ax.bar(x - width/2, model_vals, width, label='Model (Baseline)', 
                   alpha=0.8, color='steelblue', edgecolor='navy', linewidth=1.5)
    bars2 = ax.bar(x + width/2, tta_vals, width, label='TTA (+5-crop+flip)', 
                   alpha=0.8, color='darkorange', edgecolor='orangered', linewidth=1.5)
    
    # Customize
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add improvement annotation
    improvements = [(tta_vals[i] - model_vals[i]) for i in range(len(datasets))]
    for i, imp in enumerate(improvements):
        color = 'green' if imp >= 0 else 'red'
        symbol = '↑' if imp >= 0 else '↓'
        ax.text(i, 0.05, f'{symbol}{abs(imp):+.4f}', ha='center', fontsize=9,
                color=color, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_metrics_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: 01_metrics_comparison.png")
plt.close()

# ============================================================================
# 2. AUC FOCUS - CROSS-DATASET
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

datasets_extended = ['cdfv2', 'dfdc', 'Average AUC']
x = np.arange(len(datasets_extended))
width = 0.35

model_aucs = [
    model_metrics['cdfv2']['AUC'],
    model_metrics['dfdc']['AUC'],
    model_metrics['avg_auc']
]
tta_aucs = [
    tta_metrics['cdfv2']['AUC'],
    tta_metrics['dfdc']['AUC'],
    tta_metrics['avg_auc']
]

bars1 = ax.bar(x - width/2, model_aucs, width, label='Model (Baseline)', 
               alpha=0.8, color='#2E86AB', edgecolor='#1A3D54', linewidth=2)
bars2 = ax.bar(x + width/2, tta_aucs, width, label='TTA (+5-crop+flip)', 
               alpha=0.8, color='#A23B72', edgecolor='#5F1841', linewidth=2)

ax.set_ylabel('AUC Score', fontsize=13, fontweight='bold')
ax.set_title('DRF v2 AUC Performance: Model vs TTA', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets_extended, fontsize=12)
ax.legend(fontsize=12, loc='lower right', framealpha=0.95)
ax.set_ylim([0, 0.95])
ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)

# Add value labels and improvements
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    
    ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
            f'{height1:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
            f'{height2:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Improvement
    imp = height2 - height1
    imp_pct = (imp / height1 * 100) if height1 > 0 else 0
    color = '#00AA00' if imp >= 0 else '#CC0000'
    symbol = '↑' if imp >= 0 else '↓'
    
    mid_x = (bar1.get_x() + bar2.get_x() + bar1.get_width()) / 2
    ax.text(mid_x, max(height1, height2) + 0.05, 
            f'{symbol} {imp:+.4f}\n({imp_pct:+.2f}%)',
            ha='center', fontsize=10, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor=color, linewidth=1.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_auc_focus.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: 02_auc_focus.png")
plt.close()

# ============================================================================
# 3. ERROR RATES (EER & Errors)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Error Analysis: Model vs TTA', fontsize=15, fontweight='bold')

# EER Comparison
ax = axes[0]
x = np.arange(len(datasets))
width = 0.35

model_eer = [model_metrics['cdfv2']['EER'], model_metrics['dfdc']['EER']]
tta_eer = [tta_metrics['cdfv2']['EER'], tta_metrics['dfdc']['EER']]

bars1 = ax.bar(x - width/2, model_eer, width, label='Model', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, tta_eer, width, label='TTA', alpha=0.8, color='darkorange')

ax.set_ylabel('EER (%)', fontsize=12, fontweight='bold')
ax.set_title('Equal Error Rate (Lower is Better)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Top-1 Error Rate (1 - ACC)
ax = axes[1]
model_err = [1 - model_metrics['cdfv2']['ACC'], 1 - model_metrics['dfdc']['ACC']]
tta_err = [1 - tta_metrics['cdfv2']['ACC'], 1 - tta_metrics['dfdc']['ACC']]

model_err = [e * 100 for e in model_err]
tta_err = [e * 100 for e in tta_err]

bars1 = ax.bar(x - width/2, model_err, width, label='Model', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, tta_err, width, label='TTA', alpha=0.8, color='darkorange')

ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Top-1 Error Rate: 1 - Accuracy (Lower is Better)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_error_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: 03_error_analysis.png")
plt.close()

# ============================================================================
# 4. TTA IMPROVEMENT HEATMAP & SUMMARY
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('TTA Improvement Analysis', fontsize=15, fontweight='bold')

# Improvement heatmap
ax = axes[0]
metrics_for_heatmap = ["AUC", "AP", "ACC", "F1"]
improvement_matrix = np.array([
    [(tta_metrics['cdfv2'][m] - model_metrics['cdfv2'][m]) * 100 for m in metrics_for_heatmap],
    [(tta_metrics['dfdc'][m] - model_metrics['dfdc'][m]) * 100 for m in metrics_for_heatmap],
])

sns.heatmap(improvement_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            xticklabels=metrics_for_heatmap, yticklabels=['cdfv2', 'dfdc'],
            cbar_kws={'label': 'Improvement (%)'}, ax=ax, linewidths=1, linecolor='gray',
            vmin=-2, vmax=2)
ax.set_title('TTA vs Model: Percentage Change (%)', fontsize=13, fontweight='bold')

# Summary bar chart of key improvements
ax = axes[1]
improvements = {
    'cdfv2 AUC': (tta_metrics['cdfv2']['AUC'] - model_metrics['cdfv2']['AUC']),
    'dfdc AUC': (tta_metrics['dfdc']['AUC'] - model_metrics['dfdc']['AUC']),
    'Avg AUC': (tta_metrics['avg_auc'] - model_metrics['avg_auc']),
    'cdfv2 AP': (tta_metrics['cdfv2']['AP'] - model_metrics['cdfv2']['AP']),
    'dfdc AP': (tta_metrics['dfdc']['AP'] - model_metrics['dfdc']['AP']),
}

colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in improvements.values()]
bars = ax.barh(list(improvements.keys()), list(improvements.values()), color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Absolute Improvement', fontsize=12, fontweight='bold')
ax.set_title('TTA Improvements: Absolute Values', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, improvements.values()):
    x_pos = val + (0.0005 if val >= 0 else -0.0005)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
            f'{val:+.4f}', ha='left' if val >= 0 else 'right', va='center',
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_tta_improvements.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_tta_improvements.png")
plt.close()

# ============================================================================
# 5. SUMMARY TABLE as IMAGE
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

# Create summary table
summary_data = [
    ['Metric', 'cdfv2 (Model)', 'cdfv2 (TTA)', 'Change %', 'dfdc (Model)', 'dfdc (TTA)', 'Change %'],
]

for metric in ["AUC", "AP", "ACC", "F1", "EER"]:
    m_cdfv2 = model_metrics['cdfv2'][metric]
    t_cdfv2 = tta_metrics['cdfv2'][metric]
    m_dfdc = model_metrics['dfdc'][metric]
    t_dfdc = tta_metrics['dfdc'][metric]
    
    if metric == "EER":  # Lower is better for EER
        chg_cdfv2 = (t_cdfv2 - m_cdfv2)
        chg_dfdc = (t_dfdc - m_dfdc)
    else:
        chg_cdfv2 = ((t_cdfv2 - m_cdfv2) / m_cdfv2 * 100) if m_cdfv2 != 0 else 0
        chg_dfdc = ((t_dfdc - m_dfdc) / m_dfdc * 100) if m_dfdc != 0 else 0
    
    summary_data.append([
        metric,
        f'{m_cdfv2:.4f}',
        f'{t_cdfv2:.4f}',
        f'{chg_cdfv2:+.2f}%' if metric != "EER" else f'{chg_cdfv2:+.2f}%',
        f'{m_dfdc:.4f}',
        f'{t_dfdc:.4f}',
        f'{chg_dfdc:+.2f}%' if metric != "EER" else f'{chg_dfdc:+.2f}%',
    ])

summary_data.append([
    'Avg AUC',
    f'{model_metrics["avg_auc"]:.4f}',
    f'{tta_metrics["avg_auc"]:.4f}',
    f'{((tta_metrics["avg_auc"] - model_metrics["avg_auc"]) / model_metrics["avg_auc"] * 100):+.2f}%',
    '—', '—', '—'
])

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.12, 0.13, 0.13, 0.12, 0.13, 0.13, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(len(summary_data[0])):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(summary_data)):
    for j in range(len(summary_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
        table[(i, j)].set_text_props(weight='bold' if j in [0, 3, 6] else 'normal')

plt.title('Comprehensive Performance Summary: Model vs TTA', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_summary_table.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: 05_summary_table.png")
plt.close()

# ============================================================================
# Print console summary
# ============================================================================
print("\n" + "="*80)
print("📊 EVALUATION SUMMARY - MODEL vs TTA")
print("="*80)
print(f"\n{'Metric':<12} {'cdfv2 Model':>14} {'cdfv2 TTA':>14} {'dfdc Model':>14} {'dfdc TTA':>14}")
print("-" * 80)

for metric in ["AUC", "AP", "ACC", "F1", "EER"]:
    print(f"{metric:<12} {model_metrics['cdfv2'][metric]:>14.4f} {tta_metrics['cdfv2'][metric]:>14.4f} "
          f"{model_metrics['dfdc'][metric]:>14.4f} {tta_metrics['dfdc'][metric]:>14.4f}")

print("-" * 80)
print(f"{'Avg AUC':<12} {model_metrics['avg_auc']:>14.4f} {tta_metrics['avg_auc']:>14.4f}")
print("="*80)

print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}/")
print(f"   📊 01_metrics_comparison.png - Main metrics across datasets")
print(f"   📊 02_auc_focus.png - Detailed AUC comparison with improvements")
print(f"   📊 03_error_analysis.png - EER and error rates")
print(f"   📊 04_tta_improvements.png - Improvement heatmap and summary")
print(f"   📊 05_summary_table.png - Comprehensive metrics table")
