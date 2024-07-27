import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

pendulum_length = 1.0  # length of the pendulum, in meters
pendulum_dt = 0.05  # time step
cartpole_length = 0.5  # length of the pole
cartpole_dt = 0.02  # time step

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
        x = pendulum_length * np.sin(theta)
        y = -pendulum_length * np.cos(theta)
        self.line.set_data([0, x], [0, y])
        self.time_text.set_text(f'Time = {i*pendulum_dt:.2f}s')
        return self.line, self.time_text
    
    def visualize(self):
        _ = FuncAnimation(self.fig, self.animate, frames=len(self.all_thetas), interval=pendulum_dt*1000, blit=True)
        plt.show()


class CartPoleVisualizer:
    def __init__(self, all_states):
        self.all_states = all_states
        self.setup()

    def setup(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2.4, 2.4)
        self.ax.set_ylim(-0.5, 1.5)
        self.cart, = self.ax.plot([], [], 'k-', lw=5)
        self.pole, = self.ax.plot([], [], 'r-', lw=2)
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

        self.cart.set_data([], [])
        self.pole.set_data([], [])
        self.time_text.set_text('')

    # Update animation
    def animate(self, i):
        global state
        
        x, theta = self.all_states[i]
        
        cart_x = [x - 0.2, x + 0.2]
        cart_y = [0, 0]
        
        pole_x = [x, x + cartpole_length * np.sin(theta)]
        pole_y = [0, cartpole_length * np.cos(theta)]
        
        self.cart.set_data(cart_x, cart_y)
        self.pole.set_data(pole_x, pole_y)
        self.time_text.set_text(f'Time = {i * cartpole_dt:.2f}s')
        return self.cart, self.pole, self.time_text

    def visualize(self):
        _ = FuncAnimation(self.fig, self.animate, frames=len(self.all_states), interval=cartpole_dt * 1000, blit=True)
        plt.show()