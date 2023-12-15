import numpy as np
import random
import matplotlib.pyplot as plt
from math import *
from scipy import stats
from tabulate import tabulate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd

#Initialisation and data extraction
CID = 1731809
random.seed(CID)

df = pd.read_csv('sb2719.csv')
X = list(df['X']); Y = list(df['Y'])
n = len(Y)
#---------------------------------------------------------------------------------------------------------------------#


#Q 1a - histogram and boxplot
def plot_histogram(Y, labels=['Histogram of Wavelengths', 'Wavelengths(nm)', 'Frequency']):
    q1, q3 = np.quantile(Y, [0.25, 0.75])
    iqr = q3 - q1
    bin_width = 2*iqr/(n**(1/3))
    bins = ceil((max(Y) - min(Y))/bin_width)

    title, x_label, y_label = labels

    fig, ax = plt.subplots()
    ax.hist(Y, bins=bins, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def plot_box(Y, labels=['Box Plot of Wavelengths', '', 'Wavelengths(nm)']):
    title, x_label, y_label = labels
    fig, ax = plt.subplots()
    ax.boxplot(Y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def Q1a():
    plot_histogram(Y)
    plot_box(Y)
    plt.show()

#Q 1b - mean, 10% trimmed mean, median, std, iqr
def Q1b(show=True):
    table = []
    table.append(["mean", np.mean(Y)])
    table.append(["trimmed_mean", stats.trim_mean(Y, 0.1)])
    table.append(["median",np.median(Y)])
    table.append(["std", np.std(Y)])
    table.append(["interquartile range", stats.iqr(Y)])
    if show: print(tabulate(table, headers=["Statistic", "Value"], tablefmt="outline", numalign="right"))
    return dict(table)
population_stats = Q1b(show=False)

# Q 1c - Scatterplot
def Q1c():
    fig, ax = plt.subplots()
    ax.scatter(X, Y, color="black", s=1)
    ax.set_title('Wavelength VS Time Index')
    ax.set_ylabel('Wavelength(nm)')
    ax.set_xlabel('Time Index')
    plt.show()
#---------------------------------------------------------------------------------------------------------------------#


#Helper Functions
def scatter(labels=['Wavelength VS Time Index', 'Wavelength(nm)', 'Time Index']):
    plt.scatter(X, Y, s=1, color='black')
    title, xlabel, ylabel = labels

    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)

def polynomial_regression(degree, plot=True, y_vals=[], color='blue'):
    y = np.array(Y) if not len(y_vals) else np.array(y_vals)
    x = np.array(X)

    if len(y) != len(x): raise Exception("length of x and y vals must be same")

    X_poly =  PolynomialFeatures(degree).fit_transform(x.reshape(-1, 1))
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)

    y_predicted = poly_reg.predict(X_poly)
    
    if plot:
        plt.plot(x, y_predicted, color=f'{color}', label=f'degree {degree} regression')
        plt.legend()

    return y_predicted
#---------------------------------------------------------------------------------------------------------------------#


# Q 2a - plot Linear Regression
def Q2a():
    scatter()
    polynomial_regression(degree=1)
    plt.show()

# Q 2b - plot both Linear and Quadratic Regression
def Q2b():
    scatter()
    polynomial_regression(degree=1, color='blue')
    polynomial_regression(degree=2, color='red')
    plt.show()

# Q 2c - standardise X and fit higher order polynomials
x_mean = np.mean(X); x_std = np.std(X)
X = [(x - x_mean)/x_std for x in X] #Standardise X

def Q2c():
    for d in range(15, 16):
        scatter()
        polynomial_regression(degree=d)
        plt.show()

# Q 2d - AIC test on models with different degrees of regression
def AIC(degree, y_vals):
    y_predicted = polynomial_regression(degree, plot=False)
    params = degree + 1 #no. of coefficients being found in regression
    
    rss = sum([(y_vals[i] - y_predicted[i])**2 for i in range(n)]) #residual sum of squares
    sigma_sq = rss/n

    log_likelihood = -n/2 * np.log(2*pi) - n/2 * np.log(sigma_sq) - 1/(2*sigma_sq) * rss
    aic = 2*params - 2*log_likelihood
    return aic

def Q2d(show=True, start=1, end=21):
    aic_table = []
    for degree in range(start, end):
        aic_table.append([degree, AIC(degree, y_vals=Y)])
    if show:
        print(tabulate(aic_table, headers=["Degree of Regression", "AIC Value"], tablefmt="outline", numalign="right"))
        plt.plot(list(range(start, end)), list(map(lambda x: x[1], aic_table)))
        plt.scatter(list(range(start, end)), list(map(lambda x: x[1], aic_table)), s=10, color='black')
        plt.title('AIC test on models with varying degree of regression')
        plt.xlabel('Degree of regression')
        plt.ylabel('AIC score')
        plt.legend()
        plt.show()

    best_degree = min(aic_table, key=lambda x: x[1])[0] #choose the regression model with lowest AIC value
    return best_degree
best_degree = Q2d(False)

# Q 2e - residual plots
def plot_QQ(vals, labels=['title', 'x', 'y']):
    fig, ax = plt.subplots()

    observed_quantiles = (vals - np.mean(vals))/np.std(vals) #standardise the observation
    theoretical_quantiles = np.array([stats.norm.ppf((i+1)/(len(vals)+1)) for i in range(len(vals))]) #theoretical standard normal quantile

    ax.plot(theoretical_quantiles, observed_quantiles, 'o')
    ax.plot([min((theoretical_quantiles.min(), observed_quantiles.min())), max((theoretical_quantiles.max(), observed_quantiles.max()))],
            [min((theoretical_quantiles.min(), observed_quantiles.min())), max((theoretical_quantiles.max(), observed_quantiles.max()))], 'r-')
    
    title, x_label, y_label = labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def Q2e():
    residuals = np.sort(np.array(Y) - polynomial_regression(best_degree, False, Y))
    #histogram plot
    plot_histogram(residuals, labels=['Histogram of Residuals', 'Normalised Time', 'Residuals'])
    #QQ plot
    plot_QQ(residuals, labels=['QQ Plot of Residuals', 'Theoretical Quantiles (Normal Distribution)', 'Observed Quantiles'])
    plt.show()

# Q 2f - extract and fit to 85 values (10, 20, 30, ... , 850)
X = [X[i] for i in range(9, 850, 10)]
Y = [Y[i] for i in range(9, 850, 10)]

def Q2f():
    scatter(labels=['Wavelength vs Time', 'Wavelength(nm)', 'Normalised Time'])
    polynomial_regression(degree=best_degree)
    plt.show()
#---------------------------------------------------------------------------------------------------------------------#


# Q 3a & 3b - bootstrapped confidence intervals and actual confidence intervals
def Q3(n_bootstraps=10, degree=best_degree, confidence=0.95):
    y_predicted = polynomial_regression(degree=degree)
    residuals = np.array(Y) - y_predicted

    bootstrap_data = []
    for _ in range(n_bootstraps):
        boot_residual = np.random.choice(residuals, size=len(residuals))
        boot_y = boot_residual + y_predicted
        predicted_boot_y = polynomial_regression(degree=degree, plot=False, y_vals=boot_y)
        bootstrap_data.append(predicted_boot_y)
    bootstrap_data = np.array(bootstrap_data)
        
    #bootstrap confidence interval
    upper_ci = []; lower_ci = []
    for i in range(len(Y)):
        lower, upper = np.quantile(bootstrap_data[:, i], [(1-confidence)/2, (1+confidence)/2])
        lower_ci.append(lower); upper_ci.append(upper)

    #actual confidence intervals
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha/2)
    half_band = z_score*population_stats['std']/sqrt(len(Y)) #stdandard devion of true population has to be used!!

    plt.scatter(X, Y, alpha=0.5, color='black', s=10)
    plt.title('Wavelength vs Time')
    plt.xlabel('Normalised Time')
    plt.ylabel('Wavelength(nm)')
    plt.fill_between(X, lower_ci, upper_ci, alpha=0.2, color='black', label=f'{n_bootstraps} Bootstrapps 95% CI')
    plt.fill_between(X, y_predicted + half_band, y_predicted - half_band, alpha=0.2, color='orange', label='actual confidence')
    plt.legend()
    plt.show()


