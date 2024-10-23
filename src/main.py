import numpy as np
import matplotlib.pyplot as plt
import os

def plot_time_series(r_values, S0, t_max=50, num_points=1000):
    """
    Plots the solution S(t) = S0 * exp(r * t) for multiple r values.

    Parameters:
    - r_values: List of rate constants
    - S0: Initial value of S at t=0
    - t_max: Maximum time value
    - num_points: Number of points in the plot
    """
    t = np.linspace(0, t_max, num_points)
    plt.figure(figsize=(12, 8))

    for r in r_values:
        S = S0 * np.exp(r * t)
        label = f'r = {r}'
        plt.plot(t, S, label=label)

    plt.xlabel('Time t')
    plt.ylabel('Asset Price S(t)')
    plt.title('Time Series Plot of dS/dt = r S for Various r Values')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)  # Ensures the y-axis starts at 0
    plt.xlim(0, t_max)
    plt.tight_layout()

    # Save the plot
    plot_filename = 'time_series_plot.png'
    plt.savefig(plot_filename)
    plt.show()
    print(f"Plot saved as {plot_filename}")

def simulate_gbm(S0, mu, sigma, t_max=50, num_steps=1000, num_paths=5):
    """
    Simulates and plots Geometric Brownian Motion paths.

    Parameters:
    - S0: Initial asset price
    - mu: Drift coefficient
    - sigma: Volatility coefficient
    - t_max: Maximum time value
    - num_steps: Number of time steps
    - num_paths: Number of GBM paths to simulate
    """
    dt = t_max / num_steps
    t = np.linspace(0, t_max, num_steps + 1)
    plt.figure(figsize=(12, 8))

    for i in range(num_paths):
        # Generate random shocks
        W = np.random.standard_normal(size=num_steps)
        W = np.insert(W, 0, 0)  # Insert W_0 = 0
        W = np.cumsum(W) * np.sqrt(dt)  # Cumulative sum to simulate Brownian motion

        # GBM formula
        S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

        plt.plot(t, S, label=f'Path {i+1}')

    plt.xlabel('Time t')
    plt.ylabel('Asset Price S(t)')
    plt.title('Geometric Brownian Motion Simulation')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)  # Ensures the y-axis starts at 0
    plt.xlim(0, t_max)
    plt.tight_layout()

    # Save the plot
    plot_filename = 'gbm_simulation.png'
    plt.savefig(plot_filename)
    plt.show()
    print(f"GBM simulation plot saved as {plot_filename}")

if __name__ == "__main__":
    # Parameters for ODE Time Series Plot
    r_values = [0.05, -0.05, 0.1, -0.1]  # Different r values for plotting
    S0 = 10                              # Initial asset price for ODE
    t_max_ode = 50                       # Extended time range for ODE

    # Plot deterministic ODE solutions
    plot_time_series(r_values, S0, t_max_ode)

    # Parameters for Geometric Brownian Motion Simulation
    mu = 0.1      # Drift coefficient
    sigma = 0.2   # Volatility coefficient
    t_max_gbm = 50
    num_steps = 1000
    num_paths = 5

    # Simulate and plot GBM
    simulate_gbm(S0, mu, sigma, t_max_gbm, num_steps, num_paths)