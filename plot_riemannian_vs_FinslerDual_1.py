import numpy as np
import matplotlib.pyplot as plt

# Set dark background theme for clear high-contrast rendering
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#0b0f19')
ax1.set_facecolor('#0b0f19')
ax2.set_facecolor('#0b0f19')

# --- PANEL A: Single-Step Loss Reduction Across Arms ---
arms = ['AdamW\n(Baseline)', 'Sign Fresh\n(Finsler)', 'Sign Stale\n(Lag 1)', 'Uniform\n(Control)']
loss_red = [0.0219, 0.0438, 0.0412, -0.00003]  # Values as positive loss reduction
colors = ['#00f3ff', '#ff007f', '#e040fb', '#808080']

bars = ax1.bar(arms, loss_red, color=colors, width=0.55, edgecolor='white', linewidth=0.8)
ax1.set_ylabel('Loss Reduction $-\Delta L$ (Matched Norm)', color='white', fontsize=11)
ax1.set_title('A: Instantaneous Ablation at Matched Norm', color='white', fontsize=12, fontweight='bold', pad=12)
ax1.grid(axis='y', linestyle='--', alpha=0.2)

# Annotate values over bars
for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0015, f'{yval:.4f}', 
                 ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    else:
        ax1.text(bar.get_x() + bar.get_width()/2.0, 0.001, '0.0000\n(No Effect)', 
                 ha='center', va='bottom', color='#a0a0a0', fontsize=8)

ax1.set_ylim(-0.005, 0.052)

# --- PANEL B: The AdamW Stalling Gap Over Steps ---
steps = np.array([40, 70, 100, 130, 160])
ratio = np.array([1.2, 4.5, 12.1, 38.0, 75.0])  # Sign / AdamW efficiency ratio

ax2.plot(steps, ratio, marker='o', color='#ff007f', linewidth=2.5, markersize=8, label='Sign / AdamW Efficiency Ratio')
ax2.axhline(1.0, color='#00f3ff', linestyle='--', label='AdamW Baseline (1.0x)')

ax2.set_xlabel('Training Step', color='white', fontsize=11)
ax2.set_ylabel('Relative Efficiency Ratio ($\Delta L_{\mathrm{sign}} / \Delta L_{\mathrm{Adam}}$)', color='white', fontsize=11)
ax2.set_title('B: Late-Training Stalling Gap Progression', color='white', fontsize=12, fontweight='bold', pad=12)
ax2.grid(True, linestyle='--', alpha=0.2)

# Annotate key steps
ax2.annotate('Step 40: 1.2x\n(Near Parity)', xy=(40, 1.2), xytext=(45, 15),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=6),
             color='white', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#0b0f19', edgecolor='#00f3ff'))

ax2.annotate('Step 160: 75x\n(AdamW Stalls)', xy=(160, 75), xytext=(120, 65),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=6),
             color='#ff007f', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#0b0f19', edgecolor='#ff007f'))

ax2.legend(facecolor='#0b0f19', edgecolor='white', labelcolor='white', fontsize=9)

plt.tight_layout()
plt.show()
