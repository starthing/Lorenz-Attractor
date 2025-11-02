import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

sigma, rho, beta = 10, 28, 8 / 3

def lorenz(x, y, z, dt):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return x + dx * dt, y + dy * dt, z + dz * dt


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.axis('off')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)

x, y, z = 0.0, 1.0, 1.05
segments = []
colors = []

initial_segment = np.array([[x, y, z], [x, y, z]])
segments.append(initial_segment)
colors.append((0, 0.5, 0.5, 1))

line_collection = Line3DCollection(segments, linewidth=2.5)
ax.add_collection3d(line_collection)

# --- Parameters ---
dt = 0.01
steps_per_frame = 7
fade_length = 400
hue = 0.0


def update(frame):
    global x, y, z, segments, colors, hue

    # Simulate Lorenz system
    for _ in range(steps_per_frame):
        x_new, y_new, z_new = lorenz(x, y, z, dt)
        new_segment = np.array([[x, y, z], [x_new, y_new, z_new]])
        segments.append(new_segment)

        color = cm.nipy_spectral(hue % 1.0)  # smooth gradient color
        colors.append(color)
        hue += 0.002  # slower hue shift = smoother

        x, y, z = x_new, y_new, z_new

    # Keep last fade_length segments
    if len(segments) > fade_length:
        segments = segments[-fade_length:]
        colors = colors[-fade_length:]

    # Apply fading transparency
    n = len(colors)
    faded_colors = [
        (c[0], c[1], c[2], (i + 1) / n) for i, c in enumerate(colors)
    ]

    # Update the line collection
    line_collection.set_segments(segments)
    line_collection.set_color(faded_colors)

    # Rotate camera for cinematic effect
    ax.view_init(30, frame * 0.5)

    return line_collection,


ani = animation.FuncAnimation(
    fig, update, frames=1500, interval=1, blit=False
)

# --- Optional: Save video (requires ffmpeg) ---
# ani.save("lorenz_attractor.mp4", fps=60, dpi=200, codec='libx264')

plt.show()
