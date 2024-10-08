import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, beta, gamma , probplot

def fit_graph(data,dist_type,name):

    
    plt.hist(data['value'], bins=30, density=True, alpha=0.6, color='g')

    # Fit the data to a normal distribution
    if dist_type == norm:
        
        mu, std = norm.fit(data['value'])
        
        # Print the estimated parameters
        print(f"Estimated mean (mu): {mu}")
        print(f"Estimated standard deviation (std): {std}")
            # Plot the fitted PDF
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2,color="b")

        plt.title(f"Normal Fit results: mu = {mu:.2f}, std = {std:.2f}")

    elif dist_type == beta:
        # Fit the data to a beta distribution
        a, b, loc, scale = beta.fit(data['value'])

        # Print the estimated parameters
        print(f"Estimated a: {a}")
        print(f"Estimated b: {b}")
        print(f"Estimated loc: {loc}")
        print(f"Estimated scale: {scale}")

        # Plot the fitted PDF
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = beta.pdf(x, a, b, loc, scale)
        plt.plot(x, p, 'k', linewidth=2)

        plt.title(f"Beta Fit results: a = {a:.2f}, b = {b:.2f}, loc = {loc:.2f}, scale = {scale:.2f}")
     
    elif dist_type == gamma:
        # Fit the data to a gamma distribution
        shape, loc, scale = gamma.fit(data['value'])

        # Print the estimated parameters
        print(f"Estimated shape: {shape}")
        print(f"Estimated loc: {loc}")
        print(f"Estimated scale: {scale}")

        # Plot the histogram of the data and the fitted PDF


        # Plot the fitted PDF
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = gamma.pdf(x, shape, loc, scale)
        plt.plot(x, p, 'k', linewidth=2,color='r')

        plt.title(f"Gamma Fit results: shape = {shape:.2f}, loc = {loc:.2f}, scale = {scale:.2f}")


    plt.savefig(f"{name}.png")
    plt.clf()

def findBest(data,name):

    norm_mu, norm_std = norm.fit(data['value'])
    gamma_shape, gamma_loc, gamma_scale = gamma.fit(data['value'])
    beta_a, beta_b, beta_loc, beta_scale = beta.fit(data['value'])

    # Plot the histogram with fitted PDFs
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data')

    # Plot PDF for each fitted distribution
    x = np.linspace(min(data['value']), max(data['value']), 100)
    pdf_beta = beta.pdf(x, beta_a, beta_b, beta_loc, beta_scale)
    pdf_gamma = gamma.pdf(x, gamma_shape, gamma_loc, gamma_scale)
    pdf_norm = norm.pdf(x, norm_mu, norm_std)

    plt.plot(x, pdf_beta, 'r-', label='Beta')
    plt.plot(x, pdf_gamma, 'b-', label='Gamma')
    plt.plot(x, pdf_norm, 'k-', label='Normal')

    plt.legend()
    plt.title('Fitted PDFs vs Data')
    plt.show()

    # Generate QQ-plots for each distribution
    plt.figure(figsize=(12, 4))

    # QQ-plot for beta distribution
    plt.subplot(1, 3, 1)
    probplot(data['value'], dist="beta", sparams=(beta_a, beta_b, beta_loc, beta_scale), plot=plt)
    plt.title("QQ-Plot Beta")

    # QQ-plot for gamma distribution
    plt.subplot(1, 3, 2)
    probplot(data['value'], dist="gamma", sparams=(gamma_shape, gamma_loc, gamma_scale), plot=plt)
    plt.title("QQ-Plot Gamma")

    # QQ-plot for normal distribution
    plt.subplot(1, 3, 3)
    probplot(data['value'], dist="norm", sparams=(norm_mu, norm_std), plot=plt)
    plt.title("QQ-Plot Normal")

    plt.tight_layout()
    plt.savefig(f"{name}_best.png")
    plt.clf()

if __name__ == "__main__":

    stock1 = "stock1.csv"
    stock2 = "stock2-1.csv"

    df1 = pd.read_csv(stock1)
    df2 = pd.read_csv(stock2)

    
    fit_graph(df1,norm,"stock1_normal")
    fit_graph(df1,beta,"stock1_beta")
    fit_graph(df1,gamma,"stock1_gamma")

    fit_graph(df2,norm,"stock2_normal")
    fit_graph(df2,beta,"stock2_beta")
    fit_graph(df2,gamma,"stock2_gamma")


    findBest(df1,"stock1")
    findBest(df2,"stock2")

    # Fit the data to a normal distribution
    ##mu, std = norm.fit(df1['value'])



