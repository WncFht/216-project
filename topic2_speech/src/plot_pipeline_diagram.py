#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = Path('figures/upgrade/pipeline_diagram.png')
out.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.set_axis_off()
boxes = [
    (0.03, 0.55, 0.12, 0.22, 'WAV\n16 kHz'),
    (0.20, 0.55, 0.14, 0.22, 'Normalize\nSpeaker split'),
    (0.39, 0.55, 0.14, 0.22, '25 ms frames\n10 ms hop'),
    (0.58, 0.55, 0.14, 0.22, 'STFT /\nlog-mel'),
    (0.78, 0.55, 0.16, 0.22, 'Metrics\nAccuracy, F1'),
    (0.58, 0.15, 0.14, 0.22, 'MFCC + Δ + Δ²\nstatistics'),
    (0.38, 0.15, 0.14, 0.22, 'Classical\ncentroid'),
    (0.78, 0.15, 0.16, 0.22, 'AI\nMLP / CNN'),
]
for x, y, w, h, text in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.02', fc='#eef5ff', ec='#315a8a', lw=1.4))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11)

def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='->', mutation_scale=14, lw=1.3, color='#333'))
arrow((0.15,0.66),(0.20,0.66)); arrow((0.34,0.66),(0.39,0.66)); arrow((0.53,0.66),(0.58,0.66)); arrow((0.72,0.66),(0.78,0.66))
arrow((0.65,0.55),(0.65,0.37)); arrow((0.58,0.26),(0.52,0.26)); arrow((0.72,0.26),(0.78,0.26)); arrow((0.86,0.37),(0.86,0.55))
ax.text(0.5, 0.92, 'VE216 Topic 2 upgraded experimental route', ha='center', va='center', fontsize=15, fontweight='bold')
ax.text(0.5, 0.03, 'Baseline and upgrades share the same speaker-independent train/test split.', ha='center', va='bottom', fontsize=10, color='#555')
fig.tight_layout()
fig.savefig(out, dpi=220)
print(out)
