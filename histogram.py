import random as rnd
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram_sqrt_sum_squares(n, N):
    """
    Generate N samples of n independent normal variables,
    compute the sqrt of the sum of squares of z-scores for each sample,
    plot a histogram, and display mean and standard deviation.
    
    Parameters:
    n -- number of random variables per sample
    N -- number of samples
    """
    
    # Step 1: Generate N samples, each with n normal random variables
    samples = np.array([[rnd.normalvariate(1, 0.1) for _ in range(n)] for _ in range(N)])
    
    # Step 2: Compute z-scores for each column (variable)
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0, ddof=1)  # sample std
    
    z_scores = (samples - means) / stds
    
    # Step 3: Compute sqrt of sum of squares of z-scores for each sample
    sqrt_sum_squares = np.sqrt(np.sum(z_scores**2, axis=1))
    
    # Step 4: Calculate mean and standard deviation
    r_mean = np.mean(sqrt_sum_squares)
    r_std = np.std(sqrt_sum_squares, ddof=1)
    
    # Step 5: Plot histogram
    plt.hist(sqrt_sum_squares, bins=30, edgecolor='black')
    plt.xlabel("sqrt(sum of squares of z-scores)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram for variables={n}, Trials={N}")
    
    # Step 6: Add mean and std text on the plot
    plt.text(0.95, 0.95, f"Mean = {r_mean:.3f}\nStd = {r_std:.3f}", 
             transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()

# Example usage:x
plot_histogram_sqrt_sum_squares(n=6, N=255)
