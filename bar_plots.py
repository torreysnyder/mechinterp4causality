import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Conjunctive', 'Disjunctive']
values = [0.019, 0.010]
baseline = 0.005

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Create bars
bars = ax.bar(categories, values, color=['steelblue', 'coral'],
              edgecolor='black', linewidth=1.5, width=0.6)

# Add baseline as dotted red line
ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2,
           label=f'Baseline = {baseline}')

# Labels and title
ax.set_ylabel('Δp(target token)', fontsize=14)
ax.set_xlabel('Condition', fontsize=14)
ax.set_title('Impact of Corruption on Output Logits', fontsize=16, fontweight='bold')

# Add value labels on top of bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f'{value:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Set y-axis limits with some padding
ax.set_ylim(0, max(values) * 1.15)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Add legend
ax.legend(fontsize=11, loc='upper right')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('delta_p_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'delta_p_comparison.png'")

# Display the plot
plt.show()
