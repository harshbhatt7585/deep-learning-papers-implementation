import numpy as np
import matplotlib.pyplot as plt

def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ vec

# Base vector
v = np.array([1.0, 0.0])

# Example positions mapped to angles
angles = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi]

fig, ax = plt.subplots(figsize=(7, 7))

# Draw circle
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(t), np.sin(t), "--", color="gray", alpha=0.5)

# Draw rotated vectors
for i, theta in enumerate(angles):
    r = rotate(v, theta)
    ax.arrow(
        0, 0, r[0], r[1],
        head_width=0.05,
        head_length=0.08,
        length_includes_head=True,
        alpha=0.8,
        label=f"pos {i}"
    )
    ax.text(r[0] * 1.1, r[1] * 1.1, f"p{i}", fontsize=10)

ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect("equal")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("RoPE intuition: position as rotation")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()