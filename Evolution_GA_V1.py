import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random

positions = np.array([])  
wavelength = 1.0  
angles = np.linspace(-np.pi / 2, np.pi / 2, 360)  # Beam angles (in radians)

# Target point coordinates
x_target, y_target = 0, 10

def add_antenna(x, y):
    """Add an antenna at position (x, y)."""
    global positions
    if positions.size == 0:
        positions = np.array([[x, y]])
    else:
        positions = np.vstack([positions, [x, y]])
    update_plot()

def calculate_signal_propagation(theta_steer, grid_size=300):
    """Calculate signal propagation over a grid in the x, y plane."""
    global positions
    if not positions.any():
        print("No antennas to calculate signal propagation!")
        return np.zeros((grid_size, grid_size))

    x_vals = np.linspace(-50, 50, grid_size)
    y_vals = np.linspace(-50, 50, grid_size)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    k = 2 * np.pi / wavelength  # Wave number

    # Initialize field intensity
    field = np.zeros_like(x_grid, dtype=np.complex128)
    for antenna_x, antenna_y in positions:
        distances = np.sqrt((x_grid - antenna_x)**2 + (y_grid - antenna_y)**2)
        phase_shifts = -k * (antenna_x * np.sin(np.radians(theta_steer)) +
                             antenna_y * np.cos(np.radians(theta_steer)))
        field += np.exp(1j * (k * distances + phase_shifts)) / (distances**0.5 + 1e-6)

    intensity = np.abs(field)**2
    return intensity / np.max(intensity)  # Normalize

def reward_function(antenna_positions):
    """Compute the signal intensity at the target point."""
    global positions
    positions = antenna_positions  # Update global positions
    intensity = calculate_signal_propagation(theta_steer=0)  # Default to theta=0
    x_vals = np.linspace(-50, 50, intensity.shape[1])
    y_vals = np.linspace(-50, 50, intensity.shape[0])
    x_idx = np.argmin(np.abs(x_vals - x_target))
    y_idx = np.argmin(np.abs(y_vals - y_target))
    return intensity[y_idx, x_idx]

def initialize_population(size, num_antennas, x_bounds=(-20, 20), y_bounds=(0, 0)):
    """
    Initialize a population of antenna arrays with half-wavelength spacing, within specified bounds.
    """
    population = []
    half_wavelength = 0.5  # Half-wavelength spacing

    for _ in range(size):
        x_start = np.random.uniform(x_bounds[0], x_bounds[1] - (num_antennas - 1) * half_wavelength)
        x_positions = np.arange(x_start, x_start + num_antennas * half_wavelength, half_wavelength)

        # Ensure x_positions are within bounds
        x_positions = x_positions[x_positions <= x_bounds[1]]

        # Assign y-coordinates (fixed or within bounds)
        if y_bounds[0] == y_bounds[1]:
            y_positions = np.full_like(x_positions, y_bounds[0])
        else:
            y_positions = np.random.uniform(y_bounds[0], y_bounds[1], len(x_positions))

        antenna_array = np.column_stack((x_positions, y_positions))
        population.append(antenna_array)

    return population

def mutate(antenna_positions, mutation_rate=0.2):
    """Mutate antenna positions with half-wavelength constraints."""
    for i in range(len(antenna_positions)):
        if random.random() < mutation_rate:
            antenna_positions[i, 0] += random.choice([-1, 1]) * (wavelength / 2)
            antenna_positions[i, 0] = round(antenna_positions[i, 0], 1)  # Ensure grid alignment
    return antenna_positions

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    split = len(parent1) // 2
    child = np.vstack((parent1[:split], parent2[split:]))
    return child

def genetic_algorithm(num_generations, population_size, num_antennas):
    """Run the Genetic Algorithm (GA)."""
    global positions
    population = initialize_population(population_size, num_antennas)

    for generation in range(num_generations):
        # Evaluate fitness of each individual
        fitness = [reward_function(p) for p in population]

        # Select the top half of the population based on fitness (elitism)
        sorted_indices = np.argsort(fitness)[::-1]
        population = [population[i] for i in sorted_indices[:population_size // 2]]

        # Generate new population through crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        positions = population[0]  # Update global positions to the best solution so far
        update_plot()

        # Print the best fitness of the current generation
        best_fitness = max(fitness)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
        plt.pause(0.1)  # Allow visualization updates

    # Return the best solution
    best_index = np.argmax([reward_function(p) for p in population])
    return population[best_index]

def update_plot():
    """Update the antenna plot."""
    global positions
    ax[0].cla()
    ax[0].set_xlim(-10, 10)
    ax[0].set_ylim(-10, 10)
    ax[0].set_title("Antenna Array")
    ax[0].set_xlabel("X Position (wavelength)")
    ax[0].set_ylabel("Y Position (wavelength)")
    if positions.any():
        ax[0].scatter(positions[:, 0], positions[:, 1], color="blue", label="Antennas")
        ax[0].legend()
    fig.canvas.draw_idle()

def update_beam(theta_steer):
    """Update beam propagation heatmap."""
    intensity = calculate_signal_propagation(theta_steer)
    beam_propagation.set_data(intensity)
    beam_propagation.set_clim(0, 1)
    fig.canvas.draw_idle()

# Visualization setup
fig, ax = plt.subplots(2, 1, figsize=(12, 16))
fig.subplots_adjust(bottom=0.25)
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)
ax[0].set_title("Antenna Array")
ax[0].set_xlabel("X Position (wavelength)")
ax[0].set_ylabel("Y Position (wavelength)")
beam_propagation = ax[1].imshow(np.zeros((200, 200)), extent=(-10, 10, -10, 10), origin="lower", aspect="auto", cmap="hot")
ax[1].set_title("Beam Propagation")
ax[1].set_xlabel("X Position (wavelength)")
ax[1].set_ylabel("Y Position (wavelength)")
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, "Steering Angle", -90, 90, valinit=0)
slider.on_changed(lambda val: update_beam(val))

# Run the Genetic Algorithm
num_antennas = 8
best_positions = genetic_algorithm(num_generations=50, population_size=50, num_antennas=num_antennas)
positions = best_positions
update_plot()
plt.show()
