# This code is heavily based on the basic_european_numba.py that was used in class

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set a random seed to replicate results
np.random.seed(42)

# European call payoff calculation
def european_call_payoff(strike, stock_price):
    return max(stock_price - strike, 0)

# Simulate paths using Geometric Brownian Motion (beta distribution)
def simulate_price_paths_beta(initial_price, drift, volatility, beta_a, beta_b, shift, dt, T, paths):
    timesteps = int(T / dt)
    price_paths = np.zeros((paths, timesteps))
    for i in range(paths):
        current_price = initial_price
        for t in range(timesteps):
            dWt = np.random.beta(beta_a, beta_b) - shift  # Use beta distribution
            dYt = drift * dt + volatility * dWt  # Change in price
            current_price += dYt  # Update the current price
            price_paths[i, t] = current_price  # Store price in path
    return price_paths

# Monte Carlo simulation for option pricing
def monte_carlo_option_pricing(price_paths, strike, risk_free_rate, T):
    call_payoffs = np.zeros(price_paths.shape[0])
    final_prices = np.zeros(price_paths.shape[0])
    discount_factor = np.exp(-risk_free_rate * T)  # Discount factor for present value
    for i in range(price_paths.shape[0]):
        final_price = price_paths[i, -1]  # Last price in the path
        final_prices[i] = final_price
        call_payoffs[i] = european_call_payoff(strike, final_price) * discount_factor
    return final_prices, call_payoffs

def main():
    # Parameters
    paths = 5000
    initial_price = 100
    drift = 0.03
    volatility = 17.04
    dt = 1/365
    T = 1
    strike = 100
    risk_free_rate = 0.01

    # Beta distribution
    beta_a, beta_b = 9, 10
    shift = 0.35

    # Simulate price paths
    price_paths = simulate_price_paths_beta(initial_price, drift, volatility, beta_a, beta_b, shift, dt, T, paths)
    # Monte Carlo pricing
    final_prices, call_payoffs = monte_carlo_option_pricing(price_paths, strike, risk_free_rate, T)

    # Plot the set
    for i in range(min(paths, 10)):
        plt.plot(price_paths[i])
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title("Simulations of Stock Price Using Beta Distribution")
    plt.savefig('stock_price_simulation.png')
    # Results

    average_price = np.average(final_prices)
    option_price = np.average(call_payoffs)

    print(f"Average stock price after {int(1 / dt) * T} days: ${average_price:.2f}")
    print(f"Calculated European Call Option Price: ${option_price:.2f}")

if __name__ == "__main__":
    main()