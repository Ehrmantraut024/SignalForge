import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Antenna positions (4x1 linear array in X direction)
def antenna_positions_linear(num_antennas, spacing):
    x = np.linspace(-(num_antennas - 1) / 2 * spacing, (num_antennas - 1) / 2 * spacing, num_antennas)
    y = np.zeros(num_antennas)  # All antennas aligned on the x-axis
    return np.stack([x, y], axis=1)

# Beam pattern calculation for linear array
def calculate_beam_pattern_linear(angles, phases, positions):
    wavelength = 1.0  # Normalized wavelength
    k = 2 * np.pi / wavelength  # Wave number
    x = positions[:, 0]
    steering_vector = np.exp(1j * (k * x[:, None] * np.sin(angles) + phases[:, None]))
    beam_pattern = np.abs(np.sum(steering_vector, axis=0))**2
    return beam_pattern

# Heatmap update for steering
def update_beam(theta_steer):
    global heatmap_data, slider, ax, positions
    phases = -2 * np.pi * positions[:, 0] / wavelength * np.sin(np.radians(theta_steer))
    beam_pattern = calculate_beam_pattern_linear(angles, phases, positions)
    heatmap_data.set_array(np.expand_dims(beam_pattern, axis=0))  # Reshape to 2D
    fig.canvas.draw_idle()

# Parameters
num_antennas = 3
spacing = 0.5  # Wavelengths
wavelength = 1.0  # Normalized
positions = antenna_positions_linear(num_antennas, spacing)

# Beam angles
angles = np.linspace(-np.pi / 2, np.pi / 2, 360)

# Initial steering angle
theta_steer_initial = 0

# Generate beam pattern
phases = -2 * np.pi * positions[:, 0] / wavelength * np.sin(np.radians(theta_steer_initial))
beam_pattern = calculate_beam_pattern_linear(angles, phases, positions)

# Plot the antenna array and heatmap
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(bottom=0.2)

# Antenna positions
ax[0].scatter(positions[:, 0], positions[:, 1], color="blue", label="Antennas")
ax[0].set_xlim(-num_antennas / 2 * spacing - 1, num_antennas / 2 * spacing + 1)
ax[0].set_ylim(-1, 1)
ax[0].set_title("Antenna Array")
ax[0].set_xlabel("X Position (wavelength)")
ax[0].set_ylabel("Y Position (wavelength)")
ax[0].legend()

# Heatmap for the beam pattern
heatmap_data = ax[1].imshow(
    [beam_pattern],
    extent=(-90, 90, 0, 1),
    aspect="auto",
    cmap="hot",
)
ax[1].set_title("Beam Pattern")
ax[1].set_xlabel("Angle (degrees)")
ax[1].set_ylabel("Intensity")

# Add a slider for steering angle
ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, "Steering Angle", -90, 90, valinit=theta_steer_initial)

slider.on_changed(lambda val: update_beam(val))

plt.show()
