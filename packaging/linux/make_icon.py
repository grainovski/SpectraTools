"""One-off script: generate a simple icon for the AppImage using matplotlib
(avoids adding Pillow as a dependency just for icon generation)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
heights = [0.2, 0.5, 0.9, 0.6, 0.3, 0.7, 0.4]
ax.bar(range(len(heights)), heights, color="#2b7de9", width=0.8)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.axis("off")
fig.tight_layout(pad=0)
fig.savefig("packaging/linux/icon.png", dpi=100)
