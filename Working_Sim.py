import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Global variables
positions = []  # List to hold antenna positions (x, y)
wavelength = 1.0  # Normalized wavelength
angles = np.linspace(-np.pi / 2, np.pi / 2, 360)  # Beam angles (in radians)
 
def add_antenna(x, y):
    """Add an antenna at position (x, y)."""
    global positions
    positions.append([x, y])
    update_plot()

def calculate_signal_propagation(theta_steer, grid_size=300):
    """Calculate signal propagation over a grid in the x, y plane."""
    global positions
    if not positions:
        print("No antennas to calculate signal propagation!")
        return np.zeros((grid_size, grid_size))

    # Extend the grid for a larger simulation area
    x_vals = np.linspace(-50, 50, grid_size)
    y_vals = np.linspace(-50, 50, grid_size)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    positions_array = np.array(positions)
    x, y = positions_array.T
    k = 2 * np.pi / wavelength  # Wave number

    # Initialize field intensity
    field = np.zeros_like(x_grid, dtype=np.complex128)
    for antenna_x, antenna_y in positions:
        distances = np.sqrt((x_grid - antenna_x)**2 + (y_grid - antenna_y)**2)
        # Apply phase shift with steering angle
        phase_shifts = -k * (antenna_x * np.sin(np.radians(theta_steer)) +
                             antenna_y * np.cos(np.radians(theta_steer)))
        # Reduce attenuation for visibility by modifying distance scaling
        field += np.exp(1j * (k * distances + phase_shifts)) / (distances**0.5 + 1e-6)

    intensity = np.abs(field)**2
    return intensity / np.max(intensity)  # Normalize

def get_intensity_at_position(intensity_grid, x_vals, y_vals, x_pos, y_pos):
    """Retrieve the intensity at a specific (x, y) position."""
    x_idx = np.argmin(np.abs(x_vals - x_pos))  # Closest index for x position
    y_idx = np.argmin(np.abs(y_vals - y_pos))  # Closest index for y position
    return intensity_grid[y_idx, x_idx]  # Note: y is row, x is column

def intensity_at_fixed_point(theta_steer):
    """Calculate the intensity at (0, 10) every time the angle changes."""
    intensity = calculate_signal_propagation(theta_steer)
    x_vals = np.linspace(-50, 50, intensity.shape[1])  # Grid x-values
    y_vals = np.linspace(-50, 50, intensity.shape[0])  # Grid y-values
    fixed_intensity = get_intensity_at_position(intensity, x_vals, y_vals, 0.28, 20)
    print(f"Intensity at (0, 10) for angle {theta_steer}: {fixed_intensity}")


def update_beam(theta_steer):
    """Update the beam propagation pattern and beam graph based on the steering angle."""
    global beam_propagation, positions, heatmap_data, ax
    if not positions:
        print("No antennas to calculate beam pattern!")
        beam_propagation.set_data(np.zeros_like(beam_propagation.get_array()))
        heatmap_data.set_array(np.zeros((1, angles.size)))  # Reset heatmap
        fig.canvas.draw_idle()
        return

    # Update beam propagation intensity
    intensity = calculate_signal_propagation(theta_steer)
    x_vals = np.linspace(-50, 50, intensity.shape[1])  # Grid x-values
    y_vals = np.linspace(-50, 50, intensity.shape[0])  # Grid y-values

    # Update beam propagation heatmap
    beam_propagation.set_data(intensity)
    beam_propagation.set_extent([-50, 50, -50, 50])  # Adjust grid extent
    beam_propagation.set_clim(0, 1)  # Normalize color limits

    # Update beam graph
    positions_array = np.array(positions)
    x = positions_array[:, 0]
    phases = -2 * np.pi * x / wavelength * np.sin(np.radians(-theta_steer))
    beam_pattern = calculate_beam_pattern_linear(angles, phases, positions_array)
    heatmap_data.set_array(np.expand_dims(beam_pattern, axis=0))  # Ensure 2D shape
    heatmap_data.set_clim(0, np.max(beam_pattern))  # Normalize for visibility

    # Calculate intensity at fixed point
    intensity_at_fixed_point(theta_steer)

    fig.canvas.draw_idle()

def update_plot():
    """Update the antenna plot."""
    global scatter, positions
    ax[0].cla()  # Clear the previous plot
    ax[0].set_xlim(-10, 10)
    ax[0].set_ylim(-10, 10)
    ax[0].set_title("Antenna Array")
    ax[0].set_xlabel("X Position (wavelength)")
    ax[0].set_ylabel("Y Position (wavelength)")
    if positions:
        positions_array = np.array(positions)
        ax[0].scatter(positions_array[:, 0], positions_array[:, 1], color="blue", label="Antennas")
        ax[0].legend()
    fig.canvas.draw_idle()

# Beam pattern calculation for linear array
def calculate_beam_pattern_linear(angles, phases, positions):
    k = 2 * np.pi / wavelength  # Wave number
    x = positions[:, 0]
    steering_vector = np.exp(1j * (k * x[:, None] * np.sin(angles) + phases[:, None]))
    beam_pattern = np.abs(np.sum(steering_vector, axis=0))**2
    return beam_pattern

# Initial setup
fig, ax = plt.subplots(3, 1, figsize=(12, 24))  # Increase vertical size
fig.subplots_adjust(bottom=0.2)

# Top plot: Antenna positions
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)
ax[0].set_title("Antenna Array")
ax[0].set_xlabel("X Position (wavelength)")
ax[0].set_ylabel("Y Position (wavelength)")

# Middle plot: Beam propagation heatmap
beam_propagation = ax[1].imshow(
    np.zeros((200, 200)),
    extent=(-10, 10, -10, 10),
    origin="lower",
    aspect="auto",  # Make heatmap larger vertically
    cmap="hot",
)
ax[1].set_title("Beam Propagation")
ax[1].set_xlabel("X Position (wavelength)")
ax[1].set_ylabel("Y Position (wavelength)")

# Bottom plot: Beam pattern heatmap
heatmap_data = ax[2].imshow(
    [np.zeros_like(angles)],
    extent=(-90, 90, 0, 1),
    aspect="auto",
    cmap="hot",
)
ax[2].set_title("Beam Pattern")
ax[2].set_xlabel("Angle (degrees)")
ax[2].set_ylabel("Intensity")

# Add a slider for steering angle
ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, "Steering Angle", -90, 90, valinit=0)

slider.on_changed(lambda val: update_beam(-val))

# Add antennas
optimal_spacing = wavelength / 2  # Half-wavelength is the optimal spacing
num_antennas = 20  # Number of antennas
for i in range(num_antennas):
    x_position = i * optimal_spacing - (num_antennas - 1) * optimal_spacing / 2  # Centered array
    print(x_position)
    add_antenna(x_position, 0)  # All antennas placed along the x-axis

plt.show()
