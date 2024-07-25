import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

length = 1.0  # length of the pendulum, in meters
dt = 0.05  # time step

class PendulumVisualizer:
    def __init__(self, all_thetas):
        self.all_thetas = all_thetas
        self.setup()

    def setup(self):
        self.fig, ax = plt.subplots()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        self.line, = ax.plot([], [], 'o-', lw=2)
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # Animation function
    def animate(self, i):
        theta = self.all_thetas[i]
        x = length * np.sin(theta)
        y = -length * np.cos(theta)
        self.line.set_data([0, x], [0, y])
        self.time_text.set_text(f'Time = {i*dt:.2f}s')
        return self.line, self.time_text
    
    def visualize(self):
        # Creating the animation
        ani = FuncAnimation(self.fig, self.animate, frames=len(self.all_thetas), interval=dt*1000, blit=True)
        plt.show()