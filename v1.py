import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider

class BeamSteeringSimulator:
    def __init__(self, grid_size=200, freq=1.0):
        self.grid_size = grid_size
        self.freq = freq
        self.wavelength = 1 / freq
        self.grid = np.zeros((grid_size, grid_size))
        self.antennas = [(90, 100), (91, 100), (92, 100), (93, 100)]  # Optimal antenna positions
        self.phases = [0, 0, 0, 0]  # Initial phases for antennas

        self.fig, self.ax = plt.subplots()
        self.heatmap = self.ax.imshow(self.grid, origin='lower', cmap='viridis', extent=(0, grid_size, 0, grid_size))
        self.fig.colorbar(self.heatmap, ax=self.ax, label='Signal Strength')
        self.ax.set_title("Beam Steering Simulation")

        self.signal_text = TextBox(plt.axes([0.2, 0.01, 0.2, 0.05]), 'Measure at (x,y):', initial="50,50")
        self.measure_button = Button(plt.axes([0.45, 0.01, 0.1, 0.05]), 'Measure')
        self.measure_button.on_clicked(self.measure_signal)

        self.update_button = Button(plt.axes([0.6, 0.01, 0.1, 0.05]), 'Update')
        self.update_button.on_clicked(self.update_plot)

        self.angle_slider = Slider(plt.axes([0.2, 0.95, 0.65, 0.03]), 'Angle (°)', 0, 360, valinit=0)
        self.angle_slider.on_changed(self.update_phases)

        self.show_antennas()
        self.update_grid()
        plt.show()

    def calculate_signal(self, x, y):
        signal = 0
        k = 2 * np.pi / self.wavelength  # Wave number
        for (ax, ay), phase in zip(self.antennas, self.phases):
            distance = np.sqrt((x - ax)**2 + (y - ay)**2)
            phase_shift = k * distance + phase
            signal += np.cos(phase_shift)
        return signal

    def update_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.grid[x, y] = self.calculate_signal(x, y)
        self.heatmap.set_data(self.grid)
        self.heatmap.set_clim(vmin=np.min(self.grid), vmax=np.max(self.grid))
        self.fig.canvas.draw()

    def show_antennas(self):
        for ax, ay in self.antennas:
            self.ax.plot(ax + 0.5, ay + 0.5, 'ro', label="Antenna")  # Adjust to center on grid

    def measure_signal(self, event):
        coords = self.signal_text.text.split(',')
        try:
            x, y = int(coords[0]), int(coords[1])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                signal = self.calculate_signal(x, y)
                print(f"Signal strength at ({x}, {y}): {signal:.2f}")
            else:
                print("Coordinates out of bounds.")
        except ValueError:
            print("Invalid coordinates. Use format x,y")

    def update_phases(self, angle):
        angle_rad = np.deg2rad(angle)
        k = 2 * np.pi / self.wavelength
        dx = np.array([ant[0] - self.grid_size // 2 for ant in self.antennas])
        dy = np.array([ant[1] - self.grid_size // 2 for ant in self.antennas])
        self.phases = -k * (dx * np.cos(angle_rad) + dy * np.sin(angle_rad))
        self.update_grid()

    def update_plot(self, event):
        self.update_grid()

if __name__ == "__main__":
    BeamSteeringSimulator()
