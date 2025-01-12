import numpy as np
import matplotlib.pyplot as plt
import random

wavelength = 1.0  # Wavelength
x_target, y_target = 0, 10  # Target point coordinates


def calculate_signal_propagation(antenna_positions, theta_steer, grid_size=300):
    """Calculate signal propagation over a grid in the x, y plane."""
    if antenna_positions.size == 0:
        return np.zeros((grid_size, grid_size))

    x_vals = np.linspace(-50, 50, grid_size)
    y_vals = np.linspace(-50, 50, grid_size)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    k = 2 * np.pi / wavelength  # Wave number
    field = np.zeros_like(x_grid, dtype=np.complex128)

    for antenna_x, antenna_y in antenna_positions:
        distances = np.sqrt((x_grid - antenna_x) ** 2 + (y_grid - antenna_y) ** 2)
        # ϕ  phase shift
        ϕ = -k * (
            antenna_x * np.sin(np.radians(theta_steer)) +
            antenna_y * np.cos(np.radians(theta_steer))
        )
        field += np.exp(1j * (k * distances + ϕ)) / (distances**0.5 + 1e-6)

    intensity = np.abs(field) ** 2
    return intensity / np.max(intensity)


def reward_function(antenna_positions, theta_steer):
    """Compute the signal intensity at the target point."""
    intensity = calculate_signal_propagation(antenna_positions, theta_steer)
    x_vals = np.linspace(-50, 50, intensity.shape[1])
    y_vals = np.linspace(-50, 50, intensity.shape[0])
    x_idx = np.argmin(np.abs(x_vals - x_target))
    y_idx = np.argmin(np.abs(y_vals - y_target))
    return intensity[y_idx, x_idx]


def initialize_population(size, num_antennas, x_bounds=(-20, 20), y_bounds=(0, 0)):
    """Initialize a population of antenna arrays and steering angles."""
    population = []
    for _ in range(size):
        x_start = np.random.uniform(x_bounds[0], x_bounds[1] - (num_antennas - 1) * (wavelength / 2))
        x_positions = np.arange(x_start, x_start + num_antennas * (wavelength / 2), wavelength / 2)
        if len(x_positions) > num_antennas:
            x_positions = x_positions[:num_antennas]
        y_positions = np.full_like(x_positions, y_bounds[0])
        antenna_positions = np.column_stack((x_positions, y_positions))
        theta_steer = np.random.uniform(-90, 90)
        population.append((antenna_positions, theta_steer))
    return population


def mutate(individual, mutation_rate=0.5):
    """Mutate the antenna positions and beam steering angle."""
    antenna_positions, theta_steer = individual
    for i in range(len(antenna_positions)):
        if random.random() < mutation_rate:
            antenna_positions[i, 0] += random.choice([-3, 3]) * (wavelength / 2)
    if random.random() < mutation_rate:
        theta_steer += random.uniform(-10, 10)
    return antenna_positions, np.clip(theta_steer, -90, 90)


def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    positions1, angle1 = parent1
    positions2, angle2 = parent2
    split = len(positions1) // 2
    child_positions = np.vstack((positions1[:split], positions2[split:]))
    child_angle = (angle1 + angle2) / 2
    return child_positions, child_angle


def evolutionary_algorithm(num_generations, population_size, num_antennas):
    """Run the evolutionary algorithm."""
    population = initialize_population(population_size, num_antennas)
    best_solution = None
    best_fitness = -np.inf

    fitness_history = []

    for generation in range(num_generations):
        fitness = [reward_function(*individual) for individual in population]

        # Track the best solution across all generations
        gen_best_index = np.argmax(fitness)
        gen_best_fitness = fitness[gen_best_index]
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[gen_best_index]

        fitness_history.append(best_fitness)

        # Elitism: Preserve the best individual
        next_population = [best_solution]

        # Selection and reproduction
        sorted_indices = np.argsort(fitness)[::-1]
        population = [population[i] for i in sorted_indices[:population_size // 2]]

        while len(next_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = mutate(crossover(parent1, parent2))
            next_population.append(child)

        population = next_population
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    return best_solution, fitness_history


def update_plots(antenna_positions, theta_steer, fitness_history):
    """Update the antenna array, fitness history, and beam simulation."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    ax1, ax2, ax3 = axes

    # Antenna positions
    ax1.set_title("Antenna Positions")
    ax1.scatter(antenna_positions[:, 0], antenna_positions[:, 1], c='blue', label='Antennas')
    ax1.set_xlabel("X Position (wavelength)")
    ax1.set_ylabel("Y Position (wavelength)")
    ax1.legend()

    # Fitness history
    ax2.plot(fitness_history, label='Best Fitness Over Generations')
    ax2.set_title("Fitness History")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.legend()

    # Beam propagation
    intensity = calculate_signal_propagation(antenna_positions, theta_steer)
    ax3.imshow(
        intensity,
        extent=(-50, 50, -50, 50),
        origin="lower",
        aspect="auto",
        cmap="hot",
    )
    ax3.set_title("Beam Propagation Simulation")
    ax3.set_xlabel("X Position (wavelength)")
    ax3.set_ylabel("Y Position (wavelength)")

    plt.tight_layout()
    plt.show()


num_antennas = 8
best_solution, fitness_history = evolutionary_algorithm(
    num_generations=10, population_size=10, num_antennas=num_antennas
)
best_positions, best_theta = best_solution

print("Final Antenna Positions:\n", best_positions)
print("Final Beam Steering Angle:", best_theta)

update_plots(best_positions, best_theta, fitness_history)
