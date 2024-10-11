import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

def european_call_payoff(strike, stock_price):
    return max(stock_price - strike, 0)

def simulate_geometric_brownian_motion(initial_price, mu, sigma, dt, T, paths):
    timesteps = int(T / dt)
    price_paths = np.zeros((paths, timesteps))
    for i in range(paths):
        current_price = initial_price
        for t in range(timesteps):
            dWt = np.random.lognormal(mean=(mu - 0.5 * sigma**2) * dt, sigma=sigma * np.sqrt(dt)) - 1
            current_price *= np.exp(dWt)
            price_paths[i, t] = current_price
    return price_paths

def monte_carlo_basket_option_pricing(price_paths1, price_paths2, strike, risk_free_rate, scenario):
    payoffs = np.zeros(price_paths1.shape[0])
    for i in range(price_paths1.shape[0]):
        final_price1 = price_paths1[i, -1]
        final_price2 = price_paths2[i, -1]
        
        if scenario == 1:
            # Scenario 1: Outperform average
            basket_price = (final_price1 + final_price2) / 2
        elif scenario == 2:
            # Scenario 2: Outperform maximum
            basket_price = max(final_price1, final_price2)
        else:
            raise ValueError("Invalid scenario")
        
        payoffs[i] = european_call_payoff(strike, basket_price)
    
    option_price = np.mean(payoffs) * np.exp(-risk_free_rate * T)
    return option_price

data1 = pd.read_csv('stock1.csv', header=None, names=['Price'])
data2 = pd.read_csv('stock2-1.csv', header=None, names=['Price'])

log_returns1 = np.diff(np.log(data1['Price']))
log_returns2 = np.diff(np.log(data2['Price']))

mu1, sigma1 = np.mean(log_returns1), np.std(log_returns1)
mu2, sigma2 = np.mean(log_returns2), np.std(log_returns2)

paths = 10000
initial_price1 = data1['Price'].iloc[-1]
initial_price2 = data2['Price'].iloc[-1]
dt = 1/365
T = 1
strike = (initial_price1 + initial_price2) / 2  
risk_free_rate = 0.01

mu1_annual, sigma1_annual = mu1 / dt, sigma1 / np.sqrt(dt)
mu2_annual, sigma2_annual = mu2 / dt, sigma2 / np.sqrt(dt)

price_paths1 = simulate_geometric_brownian_motion(initial_price1, mu1_annual, sigma1_annual, dt, T, paths)
price_paths2 = simulate_geometric_brownian_motion(initial_price2, mu2_annual, sigma2_annual, dt, T, paths)

option_price_scenario1 = monte_carlo_basket_option_pricing(price_paths1, price_paths2, strike, risk_free_rate, 1)
option_price_scenario2 = monte_carlo_basket_option_pricing(price_paths1, price_paths2, strike, risk_free_rate, 2)

plt.figure(figsize=(12, 6))
for i in range(min(paths, 10)):
    plt.plot(price_paths1[i], color='r', alpha=0.5)
    plt.plot(price_paths2[i], color='b', alpha=0.5)
plt.xlabel('Days')
plt.ylabel('Price')
plt.title("Sample Paths of Stock Prices (Lognormal)")
plt.legend(['Stock 1', 'Stock 2'])
plt.show()

print(f"Basket Option Price (Scenario 1 - Outperform Average): ${option_price_scenario1:.2f}")
print(f"Basket Option Price (Scenario 2 - Outperform Maximum): ${option_price_scenario2:.2f}")
print(f"\nComparison: Scenario 2 price is {option_price_scenario2/option_price_scenario1:.2f} times Scenario 1 price")