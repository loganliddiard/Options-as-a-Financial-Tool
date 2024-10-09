import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, beta, lognorm , probplot , kstest

def fit_graph(data,dist_type,name):

    
    #plt.hist(data['value'], bins=30, density=True, alpha=0.6, color='g')

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
        plt.plot(x, p, 'k', linewidth=2,color="r")

        #plt.title(f"Normal Fit results: mu = {mu:.2f}, std = {std:.2f}")

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
        plt.plot(x, p, 'k', linewidth=2,color='b')

        #plt.title(f"Beta Fit results: a = {a:.2f}, b = {b:.2f}, loc = {loc:.2f}, scale = {scale:.2f}")
     
    elif dist_type == lognorm:
        shape, loc, scale = lognorm.fit(data['value'], floc=0)  # floc=0 fixes the location parameter to 0
        # Generate x values for PDF
        x = np.linspace(min(data['value']), max(data['value']), 100)

        # Plot PDF of the fitted log-normal distribution
        pdf_lognorm = lognorm.pdf(x, shape, loc=loc, scale=scale)
        plt.plot(x, pdf_lognorm, 'k', linewidth=2,color="g")


    #plt.savefig(f"{name}.png")
    #plt.clf()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, lognorm, kstest, probplot

def findBest(data, name):
    # Fit distributions to the data
    norm_mu, norm_std = norm.fit(data['value'])
    beta_a, beta_b, beta_loc, beta_scale = beta.fit(data['value'])
    ln_shape, ln_loc, ln_scale = lognorm.fit(data['value'], floc=0)  # Fit log-normal distribution

    # Plot the histogram with fitted PDFs
    plt.hist(data['value'], bins=30, density=True, alpha=0.6, color='g', label='Data')

    # Generate x values for PDFs
    x = np.linspace(min(data['value']), max(data['value']), 100)

    # Plot PDFs for each fitted distribution
    pdf_beta = beta.pdf(x, beta_a, beta_b, beta_loc, beta_scale)
    pdf_norm = norm.pdf(x, norm_mu, norm_std)
    pdf_lognorm = lognorm.pdf(x, ln_shape, loc=ln_loc, scale=ln_scale)

    plt.plot(x, pdf_beta, 'k-', label='Beta')
    plt.plot(x, pdf_norm, 'b-', label='Normal')
    plt.plot(x, pdf_lognorm, 'r-', label='Log-Normal')  # Add Log-Normal PDF

    plt.legend()
    plt.title(f'Fitted PDFs vs Data for {name}')

    # Generate QQ-plots for each distribution
    plt.figure(figsize=(12, 4))

    # QQ-plot for beta distribution
    plt.subplot(1, 3, 1)
    probplot(data['value'], dist="beta", sparams=(beta_a, beta_b, beta_loc, beta_scale), plot=plt)
    plt.title("QQ-Plot Beta")

    # QQ-plot for normal distribution
    plt.subplot(1, 3, 2)
    probplot(data['value'], dist="norm", sparams=(norm_mu, norm_std), plot=plt)
    plt.title("QQ-Plot Normal")

    # QQ-plot for log-normal distribution
    plt.subplot(1, 3, 3)
    probplot(data['value'], dist="lognorm", sparams=(ln_shape, ln_loc, ln_scale), plot=plt)
    plt.title("QQ-Plot Log-Normal")

    plt.tight_layout()
    plt.savefig(f"{name}_best.png")
    plt.clf()

    # One-sample KS test for Normal distribution
    ks_stat_norm, p_value_norm = kstest(data['value'], 'norm', args=(norm_mu, norm_std))
    print(f"KS test for Normal distribution: Statistic = {ks_stat_norm}, p-value = {p_value_norm}")

    # One-sample KS test for Beta distribution
    ks_stat_beta, p_value_beta = kstest(data['value'], 'beta', args=(beta_a, beta_b, beta_loc, beta_scale))
    print(f"KS test for Beta distribution: Statistic = {ks_stat_beta}, p-value = {p_value_beta}")

    # One-sample KS test for Log-Normal distribution
    ks_stat_lognorm, p_value_lognorm = kstest(data['value'], 'lognorm', args=(ln_shape, ln_loc, ln_scale))
    print(f"KS test for Log-Normal distribution: Statistic = {ks_stat_lognorm}, p-value = {p_value_lognorm}")


##Main function
if __name__ == "__main__":

    stock1 = "stock1.csv"
    stock2 = "stock2-1.csv"

    df1 = pd.read_csv(stock1)
    df2 = pd.read_csv(stock2)

    ##allows us to create legend of two different types of distributions we will be fitting and comparing
    handles = [plt.Line2D([], [], color='red', label='Normal'),
                plt.Line2D([], [], color='blue', label='Beta'),
                plt.Line2D([], [], color='green', label='Log-Normal')]

##Graph stock 1
    print(f"Graphing {stock1} distributions...")
    plt.hist(df1['value'], bins=30, density=True, alpha=0.6, color='g')
    plt.title(f"Stock 1 Fitted Distributions")
    fit_graph(df1,norm,"stock1_normal")
    fit_graph(df1,beta,"stock1_beta")
    fit_graph(df1,lognorm,"stock1_lognormal")

    plt.legend(handles=handles)

    plt.savefig("stock1_distributions.png")
    plt.clf()

## Graph stock 2
    print(f"Graphing {stock2} distributions...")
    plt.hist(df1['value'], bins=30, density=True, alpha=0.6, color='g')
    plt.title(f"Stock 2 Fitted Distributions")
    fit_graph(df2,norm,"stock2_normal")
    fit_graph(df2,beta,"stock2_beta")
    fit_graph(df2,lognorm,"stock2_lognormal")

    plt.legend(handles=handles)

    plt.savefig("stock2_distributions.png")
    plt.clf()

##Finding best

    print("\n---------------------------\n")
    print("testing and graphing stock 1 data...")
    findBest(df1,"Stock 1")
    print("testing and graphing stock 2 data...")
    findBest(df2,"Stock 2")