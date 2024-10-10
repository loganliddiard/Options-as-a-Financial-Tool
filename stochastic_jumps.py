import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

# Set seed to replicate results
np.random.seed(42)

# Helper function to calculate daily returns from price data
def calculate_daily_returns(prices):
    return prices.pct_change().dropna()  # Calculate percentage change and drop NaN values

# Helper function to filter out non-positive values
def filter_positive_values(returns):
    return returns[returns > 0]

# Function to calculate the payoff for a European call option
def european_call_payoff(stock_price, strike_price):
    """Calculate the payoff for a European call option."""
    return max(stock_price - strike_price, 0)

# Monte Carlo simulation using log-normal distribution
def monte_carlo_simulation_log_normal(mu, sigma, initial_price, T, dt, paths):
    timesteps = int(T / dt)
    price_paths = np.zeros((paths, timesteps))
    for i in range(paths):
        current_price = initial_price
        for t in range(timesteps):
            # Random price movement using log-normal distribution
            random_shock = np.random.lognormal(mean=mu * dt, sigma=sigma * np.sqrt(dt))
            current_price *= random_shock  # Update the price
            price_paths[i, t] = current_price  # Store the price in the path
    return price_paths

# Monte Carlo basket option pricing
def monte_carlo_basket_option_pricing(price_paths1, price_paths2, risk_free_rate, T, initial_price, scenario='average'):
    payoffs = np.zeros(price_paths1.shape[0])
    discount_factor = np.exp(-risk_free_rate * T)  # Discount factor for present value
    
    for i in range(price_paths1.shape[0]):
        # Final prices for both stocks
        final_price_stock1 = price_paths1[i, -1]
        final_price_stock2 = price_paths2[i, -1]

        # Determine the benchmark based on the scenario
        if scenario == 'average':
            # Payoff for average value of stock1 and stock2
            benchmark = (final_price_stock1 + final_price_stock2) / 2
        elif scenario == 'maximum':
            # Payoff for max value of either stock1 or stock2
            benchmark = max(final_price_stock1, final_price_stock2)

        # Calculate option payoff
        payoffs[i] = max(benchmark - initial_price, 0) * discount_factor

    return np.mean(payoffs)  # Return the average option price

# Parameters
initial_price = 100
T = 1  # 1 year
dt = 1/365  # one day each step
paths = 5000
risk_free_rate = 0.01  # Assume 1% risk-free rate

# Load stock data
stock1_data = pd.read_csv('stock1.csv')
stock2_data = pd.read_csv('stock2-1.csv')

# Calculate daily returns for both stocks
stock1_returns = calculate_daily_returns(stock1_data['value'])
stock2_returns = calculate_daily_returns(stock2_data['value'])

# Filter out non-positive returns to fit log-normal distribution
stock1_returns = filter_positive_values(stock1_returns)
stock2_returns = filter_positive_values(stock2_returns)

# Check if there are any remaining non-positive values
if len(stock1_returns) == 0 or len(stock2_returns) == 0:
    raise ValueError("The returns data contains no positive values, which are required for log-normal fitting.")

# Fit log-normal distribution to filtered daily returns of both stocks
ln_shape_stock1, ln_loc_stock1, ln_scale_stock1 = lognorm.fit(stock1_returns, floc=0)
ln_shape_stock2, ln_loc_stock2, ln_scale_stock2 = lognorm.fit(stock2_returns, floc=0)

# Calculate  mean and standard deviation for log-normal distribution
mu_stock1 = np.mean(stock1_returns)
sigma_stock1 = np.std(stock1_returns)
mu_stock2 = np.mean(stock2_returns)
sigma_stock2 = np.std(stock2_returns)

# Simulate price paths for both stocks using the log-normal distribution
price_paths_stock1 = monte_carlo_simulation_log_normal(mu_stock1, sigma_stock1, initial_price, T, dt, paths)
price_paths_stock2 = monte_carlo_simulation_log_normal(mu_stock2, sigma_stock2, initial_price, T, dt, paths)

# Scenario 1: Option pays off if outperforms average value of stock1 and stock2
option_price_average = monte_carlo_basket_option_pricing(price_paths_stock1, price_paths_stock2, risk_free_rate, T, initial_price, scenario='average')

# Scenario 2: Option pays off if outperforms max value of stock1 or stock2
option_price_maximum = monte_carlo_basket_option_pricing(price_paths_stock1, price_paths_stock2, risk_free_rate, T, initial_price, scenario='maximum')

# results
print(f"Calculated Basket Option Price (Average Scenario): ${option_price_average:.2f}")
print(f"Calculated Basket Option Price (Maximum Scenario): ${option_price_maximum:.2f}")

# Plot first 10 simulated paths for both stocks
plt.figure(figsize=(12, 6))
for i in range(min(paths, 10)):
    plt.plot(price_paths_stock1[i], color='blue', alpha=0.5)
    plt.plot(price_paths_stock2[i], color='green', alpha=0.5)
plt.xlabel('Days')
plt.ylabel('Price')
plt.title("Simulations of Stock Prices Using Log-normal Distribution")
plt.legend(['Stock 1 Paths', 'Stock 2 Paths'])
plt.grid()
plt.savefig('stock_price_stochastic_jumps.png')
