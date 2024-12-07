import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit



columns = ['algorithm', 'reduction_level', 'node_count', 'edge_count', 
           'time_taken', 'current_memory', 'peak_memory', 
           'number_of_communities', 'size_of_communities',]

# load data
df = pd.read_csv("cdavgresults.csv")


# Set up the plot style
sns.set_theme(style="whitegrid")

# Loop through each metric and plot
y_per_reduction_level= ['node_count', 'edge_count', 'time_taken', 'current_memory', 'peak_memory', 'time_taken', 'number_of_communities', 'size_of_communities',]


# Loop through each metric and plot for each y variable
for alg in df['algorithm'].unique():
    alg_data = df[df['algorithm'] == alg]  # Filter data for the current metric
    for y in y_per_reduction_level:
        plt.figure(figsize=(10, 6))
        
        color = '#32a88b'

        # Fit logarithmic regression
        X = alg_data['reduction_level'].values.reshape(-1, 1)
        log_X = np.log(X + 1)  # Add 1 to avoid log(0)
        y_values = alg_data[y].values
        model = LinearRegression().fit(log_X, y_values)
        y_pred = model.predict(log_X)
        r2 = r2_score(y_values, y_pred)
            
            # Plot data with the corresponding color
        # color = dataa[]  # Get the color for the threshold
        sns.lineplot(
            x="reduction_level",
            y=y,
            data=alg_data,
            marker="o",
            label=f"(Log R²={r2:.2f})",
            # color=color  # Apply the color to the solid line
        )
            
        # Plot the trendline with the same color but as a dashed line
        plt.plot(
            alg_data['reduction_level'], y_pred, 
            linestyle="--", color=color, label=f"Trendline"
        )
        
        # Add titles and labels
        plt.title(f"{y} across reduction levels for {alg}", fontsize=14)
        plt.xlabel("reduction level", fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.xticks(alg_data["reduction_level"].unique())
        plt.tight_layout()
        
        # Show plot
        plt.savefig(f'./visualizations/{y}_for_{alg}.png')
        #plt.show()

"""
# Ensure the directory exists for saving plots
os.makedirs('./visualizations/', exist_ok=True)

# Set up the plot style
sns.set_theme(style="whitegrid")

# Define models
def logarithmic(x, a, b):
    return a * np.log(x + 1) + b  # Add 1 to avoid log(0)

def exponential(x, a, b):
    return a * np.exp(b * x)

# Iterate through thresholds
for threshold in df['threshold'].unique():
    plt.figure(figsize=(12, 8))
    
    # Filter data for the current threshold
    threshold_data = df[df['threshold'] == threshold]
    
    # Set up a color palette for metrics
    palette = sns.color_palette("tab10", n_colors=threshold_data['metric'].nunique())
    metric_colors = dict(zip(threshold_data['metric'].unique(), palette))
    
    for metric in threshold_data['metric'].unique():
        metric_data = threshold_data[threshold_data['metric'] == metric]
        
        # Fit logarithmic and exponential models
        X = metric_data['reduction_level'].values
        y_values = metric_data['peak_memory'].values
        
        try:
            # Logarithmic fit
            log_params, _ = curve_fit(logarithmic, X, y_values)
            log_y_pred = logarithmic(X, *log_params)
            log_r2 = r2_score(y_values, log_y_pred)
            
            # Exponential fit
            exp_params, _ = curve_fit(exponential, X, y_values, maxfev=10000)
            exp_y_pred = exponential(X, *exp_params)
            exp_r2 = r2_score(y_values, exp_y_pred)
            
            # Choose the best model
            if log_r2 > exp_r2:
                best_fit = "Logarithmic"
                best_params = log_params
                best_r2 = log_r2
                best_y_pred = log_y_pred
                equation = f"{log_params[0]:.2f}*ln(x) + {log_params[1]:.2f}"
            else:
                best_fit = "Exponential"
                best_params = exp_params
                best_r2 = exp_r2
                best_y_pred = exp_y_pred
                equation = f"{exp_params[0]:.2f}*e^({exp_params[1]:.2f}*x)"
            
            # Plot data
            sns.lineplot(
                x="reduction_level",
                y="peak_memory",
                data=metric_data,
                marker="o",
                label=f"{metric} ({best_fit}, R²={best_r2:.2f})",
                color=metric_colors[metric]
            )
            
            # Add trendline
            plt.plot(
                X, best_y_pred, 
                linestyle="--", color=metric_colors[metric],
                label=f"{metric} Equation: {equation}"
            )
        
        except Exception as e:
            print(f"Could not fit models for {metric} at threshold {threshold}: {e}")
    
    # Add titles and labels
    title = f"Peak Memory Across Reduction Levels at Threshold {threshold}"
    plt.title(title, fontsize=16)
    plt.xlabel("Reduction Level", fontsize=14)
    plt.ylabel("Peak Memory (bytes)", fontsize=14)
    plt.legend(title="Metrics", loc="best", fontsize=10)
    plt.tight_layout()
    
    # Save and show the plot
    safe_title = title.replace(' ', '_').replace('/', '_')
    plt.savefig(f'./visualizations/{safe_title}.png')
    plt.show()

"""