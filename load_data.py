import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set plot style
plt.style.use('default')
sns.set_theme()

# Path to the CSV files directory
csv_dir = os.path.join(os.getcwd(), "data")  # Point to the data directory

def load_csv(filename):
    """Loads a CSV file and returns it as a DataFrame."""
    file_path = os.path.join(csv_dir, filename)
    print(f"Loading {filename}...")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        raise

# Load necessary datasets
players_df = load_csv("players.csv")
clubs_df = load_csv("clubs.csv")
player_valuations_df = load_csv("player_valuations.csv")
appearances_df = load_csv("appearances.csv")

# Display basic information about the datasets
print("\nDataset shapes:")
print(f"Players: {players_df.shape}")
print(f"Clubs: {clubs_df.shape}")
print(f"Player Valuations: {player_valuations_df.shape}")
print(f"Appearances: {appearances_df.shape}") 