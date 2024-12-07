import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

console=Console()

# Assume `df` contains your data with the specified columns
columns = [
    'time_taken', 'current_memory', 'peak_memory', 'centrality_scores', 'metric', 'trial',
    'threshold', 'reduction_level', 'node_count', 'edge_count',
    'node_retention', 'edge_retention', 'connectivity'
]

# Example DataFrame (replace this with your actual data)
df = pd.read_csv("../Desktop/all_results.csv")

grouped = df.groupby(['metric', 'threshold', 'reduction_level'])['trial'].nunique()
console.print(grouped)
# Reset index to make grouped columns normal columns again
numeric_columns = ['time_taken', 'current_memory', 'peak_memory', 'node_count', 'edge_count', 'node_retention', 'edge_retention', 'connectivity']
grouped = df.groupby(['metric', 'threshold', 'reduction_level'])[numeric_columns].mean()
averaged_df = grouped.reset_index()

# Save to CSV or print
averaged_df.to_csv("averaged_centrality_results.csv", index=False)
print(averaged_df)


# Load your data
df = pd.read_csv("averaged_centrality_results.csv")

# Set up the plot style
sns.set(style="whitegrid")

# Loop through each metric and plot node retention
y_per_reduction_level= ['node_retention', 'edge_retention', 'peak_memory', 'time_taken', 'node_count', 'edge_count']
"""
for metric in df['metric'].unique():
    metric_data = df[df['metric'] == metric]  # Filter data for the current metric
    for y in y_per_reduction_level:

	    # Plot using Seaborn
	    plt.figure(figsize=(10, 6))
	    sns.lineplot(
		data=metric_data,
		x="reduction_level",
		y=y,
		hue="threshold",
		palette="viridis",
		marker="o"
	    )

	    # Add titles and labels
	    plt.title(f"{y} Across Reduction Levels for {metric}", fontsize=14)
	    plt.xlabel("Reduction Level", fontsize=12)
	    plt.ylabel(y, fontsize=12)
	    plt.legend(title="Threshold", loc="best")
	    plt.xticks(metric_data["reduction_level"].unique())
	    plt.tight_layout()

	    # Show plot
	    plt.show()
"""

# linear
"""
for metric in df['metric'].unique():
    metric_data = df[df['metric'] == metric]  # Filter data for the current metric
    for y in y_per_reduction_level:
        plt.figure(figsize=(10, 6))
        
        # Iterate through thresholds
        for threshold in metric_data['threshold'].unique():
            threshold_data = metric_data[metric_data['threshold'] == threshold]
            
            # Fit linear regression
            X = threshold_data['reduction_level'].values.reshape(-1, 1)
            y_values = threshold_data[y].values
            model = LinearRegression().fit(X, y_values)
            y_pred = model.predict(X)
            r2 = r2_score(y_values, y_pred)
            
            # Plot data and regression line
            sns.lineplot(
                x="reduction_level",
                y=y,
                data=threshold_data,
                marker="o",
                label=f"Threshold {threshold} (R²={r2:.2f})"
            )
            plt.plot(threshold_data['reduction_level'], y_pred, linestyle="--", label=f"Trendline {threshold}")
        
        # Add titles and labels
        plt.title(f"{y} Across Reduction Levels for {metric}", fontsize=14)
        plt.xlabel("Reduction Level", fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.legend(title="Threshold", loc="best")
        plt.xticks(metric_data["reduction_level"].unique())
        plt.tight_layout()
        
        # Show plot
        plt.show()
"""
# logarithmic

# Loop through each metric and plot for each y variable
for metric in df['metric'].unique():
    metric_data = df[df['metric'] == metric]  # Filter data for the current metric
    for y in y_per_reduction_level:
        plt.figure(figsize=(10, 6))
        
        # Create the plot and retrieve color palette
        palette = sns.color_palette("viridis", n_colors=metric_data['threshold'].nunique())
        threshold_colors = dict(zip(metric_data['threshold'].unique(), palette))
        
        # Iterate through thresholds
        for threshold in metric_data['threshold'].unique():
            threshold_data = metric_data[metric_data['threshold'] == threshold]
            
            # Fit logarithmic regression
            X = threshold_data['reduction_level'].values.reshape(-1, 1)
            log_X = np.log(X + 1)  # Add 1 to avoid log(0)
            y_values = threshold_data[y].values
            model = LinearRegression().fit(log_X, y_values)
            y_pred = model.predict(log_X)
            r2 = r2_score(y_values, y_pred)
            
            # Plot data with the corresponding color
            color = threshold_colors[threshold]  # Get the color for the threshold
            sns.lineplot(
                x="reduction_level",
                y=y,
                data=threshold_data,
                marker="o",
                label=f"Threshold {threshold} (Log R²={r2:.2f})",
                color=color  # Apply the color to the solid line
            )
            
            # Plot the trendline with the same color but as a dashed line
            plt.plot(
                threshold_data['reduction_level'], y_pred, 
                linestyle="--", color=color, label=f"Trendline {threshold}"
            )
        
        # Add titles and labels
        plt.title(f"{y} Across Reduction Levels for {metric}", fontsize=14)
        plt.xlabel("Reduction Level", fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.legend(title="Threshold", loc="best")
        plt.xticks(metric_data["reduction_level"].unique())
        plt.tight_layout()
        
        # Show plot
        plt.savefig(f'./visualizations/{y}_Across_Reduction_Levels_for_{metric}.png')
        #plt.show()

"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Ensure the directory exists for saving plots
os.makedirs('./visualizations/', exist_ok=True)

# Set up the plot style
sns.set(style="whitegrid")

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
        
        # Fit linear regression to visualize trends
        X = metric_data['reduction_level'].values.reshape(-1, 1)
        y_values = metric_data['peak_memory'].values
        model = LinearRegression().fit(X, y_values)
        y_pred = model.predict(X)
        r2 = r2_score(y_values, y_pred)
        
        # Plot data
        sns.lineplot(
            x="reduction_level",
            y="peak_memory",
            data=metric_data,
            marker="o",
            label=f"{metric} (R²={r2:.2f})",
            color=metric_colors[metric]
        )
        
        # Add trendline
        plt.plot(
            metric_data['reduction_level'], y_pred, 
            linestyle="--", color=metric_colors[metric]
        )

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

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Ensure the directory exists for saving plots
os.makedirs('./visualizations/', exist_ok=True)

# Set up the plot style
sns.set(style="whitegrid")

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
                y="time_taken",
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
    title = f"Time Taken Across Reduction Levels at Threshold {threshold}"
    plt.title(title, fontsize=16)
    plt.xlabel("Reduction Level", fontsize=14)
    plt.ylabel("Time Taken (bytes)", fontsize=14)
    plt.legend(title="Metrics", loc="best", fontsize=10)
    plt.tight_layout()
    
    # Save and show the plot
    safe_title = title.replace(' ', '_').replace('/', '_')
    plt.savefig(f'./visualizations/{safe_title}.png')
    plt.show()

