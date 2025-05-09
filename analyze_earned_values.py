import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter

# Path to the CSV files directory
csv_dir = r"C:\Users\vinay\Desktop\Uni\Semester 6\Data Science in Practice\Project\csv files"

# Function to load a CSV file
def load_csv(filename):
    """
    Loads a CSV file and returns it as a DataFrame.
    
    Args:
        filename: Name of the CSV file to load
    
    Returns:
        DataFrame containing the CSV data
    """
    # Construct full path to the CSV file
    file_path = os.path.join(csv_dir, filename)
    
    # Load the CSV file
    print(f"Loading {filename}...")
    return pd.read_csv(file_path)

# Load all necessary datasets
print("=== LOADING DATASETS ===")
players_df = load_csv("players.csv")
clubs_df = load_csv("clubs.csv")
player_valuations_df = load_csv("player_valuations.csv")
appearances_df = load_csv("appearances.csv")
game_events_df = load_csv("game_events.csv")

# Print column names to debug
print("\nPlayers DataFrame columns:")
print(players_df.columns.tolist())
print("\nPlayer Valuations DataFrame columns:")
print(player_valuations_df.columns.tolist())
print("\nAppearances DataFrame columns:")
print(appearances_df.columns.tolist())

def analyze_actual_vs_earned_market_values():
    """
    Analyzes the relationship between actual market values and "earned" market values
    based on player performance statistics.
    
    The "earned" market value is what a player should be worth based on their
    performance metrics and other relevant factors.
    """
    print("\n=== ANALYZING ACTUAL VS 'EARNED' MARKET VALUES ===")
    
    # Step 1: Merge player data with their valuations
    print("Merging players with their valuations...")
    player_values = pd.merge(
        players_df,
        player_valuations_df,
        how='inner',
        on='player_id'
    )
    
    # Check the merged dataframe columns
    print("\nMerged player_values columns:")
    print(player_values.columns.tolist())
    
    # Identify the market value columns after the merge
    market_value_cols = [col for col in player_values.columns if 'market_value' in col.lower()]
    print(f"\nMarket value columns: {market_value_cols}")
    
    # Use the correct market value column from the merge
    market_value_col = 'market_value_in_eur_y' if 'market_value_in_eur_y' in player_values.columns else 'market_value_in_eur'
    print(f"Using market value column: {market_value_col}")
    
    # Step 2: Add player performance metrics from appearances
    print("Adding player performance metrics...")
    
    # Check the columns in appearances_df to ensure we have the expected columns for aggregation
    expected_columns = ['player_id', 'goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards']
    missing_columns = [col for col in expected_columns if col not in appearances_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in appearances_df: {missing_columns}")
        print("Please check the appearances data structure")
        
        # Create sample columns if they don't exist (for demonstration purposes)
        for col in missing_columns:
            if col != 'player_id':  # Skip player_id as it should exist
                appearances_df[col] = 0
    
    # Aggregate player statistics from appearances
    agg_dict = {}
    for col in ['goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards']:
        if col in appearances_df.columns:
            agg_dict[col] = 'sum'
    
    player_stats = appearances_df.groupby('player_id').agg(agg_dict).reset_index()
    
    # Merge player values with their statistics
    player_data = pd.merge(
        player_values,
        player_stats,
        how='inner',
        on='player_id'
    )
    
    # Step 3: Calculate additional metrics
    print("Calculating additional performance metrics...")
    
    # Calculate games played (approximation based on minutes)
    player_data['games_played'] = player_data['minutes_played'] / 90
    
    # Calculate goal contributions per game
    player_data['goal_contributions_per_game'] = (player_data['goals'] + player_data['assists']) / player_data['games_played'].replace(0, 1)
    
    # Calculate minutes per goal contribution
    player_data['minutes_per_goal_contribution'] = player_data['minutes_played'] / (player_data['goals'] + player_data['assists'] + 1)  # +1 to avoid division by zero
    
    # Add age as a factor (could affect market value)
    player_data['age'] = (pd.to_datetime('now') - pd.to_datetime(player_data['date_of_birth'], errors='coerce')).dt.days / 365.25
    
    # Step 4: Build a model to predict "earned" market values
    print("Building predictive model for 'earned' market values...")
    
    # Select features to use in our model
    features = [
        'age', 'height_in_cm', 'goals', 'assists', 
        'minutes_played', 'yellow_cards', 'red_cards',
        'goal_contributions_per_game', 'minutes_per_goal_contribution',
        'games_played'
    ]
    
    # Check if all features exist in our dataframe
    missing_features = [feat for feat in features if feat not in player_data.columns]
    if missing_features:
        print(f"Warning: Missing features in player_data: {missing_features}")
        # Remove missing features from the list
        features = [feat for feat in features if feat not in missing_features]
    
    # Filter out rows with missing values
    model_data = player_data.dropna(subset=features + [market_value_col])
    
    # Log transform the market value (often follows a log-normal distribution)
    model_data['log_market_value'] = np.log1p(model_data[market_value_col])
    
    # Prepare features and target
    X = model_data[features]
    y = model_data['log_market_value']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model (more robust to outliers and complex relationships)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict "earned" market values
    y_pred = model.predict(X_test_scaled)
    
    # Calculate model performance
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model performance: R² = {r2:.4f}, RMSE = {rmse:.4f}")
    
    # Step 5: Generate predictions for all players and compare with actual values
    print("Comparing actual vs. 'earned' market values...")
    
    # Prepare all data for prediction
    X_all = scaler.transform(model_data[features])
    
    # Predict earned market values for all players
    model_data['predicted_log_market_value'] = model.predict(X_all)
    model_data['predicted_market_value'] = np.expm1(model_data['predicted_log_market_value'])
    
    # Calculate the difference between actual and earned market values
    model_data['market_value_difference'] = model_data[market_value_col] - model_data['predicted_market_value']
    model_data['market_value_ratio'] = model_data[market_value_col] / model_data['predicted_market_value']
    
    # Convert values to millions for better readability
    model_data['actual_value_millions'] = model_data[market_value_col] / 1000000
    model_data['predicted_value_millions'] = model_data['predicted_market_value'] / 1000000
    model_data['value_difference_millions'] = model_data['market_value_difference'] / 1000000
    
    # Step 6: Visualize the results
    print("Creating visualizations...")
    
    # Function to format the axis labels in millions
    def millions_formatter(x, pos):
        return f'{x:.0f}M'
    
    # Plot actual vs. predicted market values
    plt.figure(figsize=(10, 8))
    plt.scatter(
        model_data['actual_value_millions'],
        model_data['predicted_value_millions'],
        alpha=0.5
    )
    plt.plot(
        [0, model_data['actual_value_millions'].max()],
        [0, model_data['actual_value_millions'].max()],
        'r--'
    )
    plt.xlabel('Actual Market Value (Millions €)')
    plt.ylabel('Predicted "Earned" Market Value (Millions €)')
    plt.title('Actual vs. "Earned" Market Values')
    
    # Format axes with millions
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    
    plt.savefig('actual_vs_earned_values.png')
    
    # Plot the most overvalued players
    overvalued = model_data.sort_values('market_value_ratio', ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x=overvalued['last_name'],
        y=overvalued['market_value_ratio'],
        palette='coolwarm'
    )
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.2,
            f'{bar.get_height():.1f}x',
            ha='center',
            color='black',
            fontsize=9
        )
        
    plt.xticks(rotation=90)
    plt.title('Top 20 Most Overvalued Players (Actual/Earned Value Ratio)')
    plt.xlabel('Player')
    plt.ylabel('Actual/Earned Value Ratio')
    plt.tight_layout()
    plt.savefig('overvalued_players.png')
    
    # Plot the most undervalued players
    undervalued = model_data.sort_values('market_value_ratio').head(20)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x=undervalued['last_name'],
        y=undervalued['market_value_ratio'],
        palette='coolwarm_r'
    )
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.02,
            f'{bar.get_height():.2f}x',
            ha='center',
            color='black',
            fontsize=9
        )
        
    plt.xticks(rotation=90)
    plt.title('Top 20 Most Undervalued Players (Actual/Earned Value Ratio)')
    plt.xlabel('Player')
    plt.ylabel('Actual/Earned Value Ratio')
    plt.tight_layout()
    plt.savefig('undervalued_players.png')
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        x=feature_importance['Importance'],
        y=feature_importance['Feature'],
        palette='viridis'
    )
    
    # Add value labels at the end of bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.3f}',
            va='center',
            color='black',
            fontsize=9
        )
        
    plt.title('Feature Importance for Predicting Market Value')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("\nCharts have been saved to the current directory.")
    
    # Step 7: Save the most interesting results
    print("Saving results to CSV files...")
    
    # Save data about overvalued and undervalued players
    overvalued_players = model_data.sort_values('market_value_difference', ascending=False).head(100)
    overvalued_players[['first_name', 'last_name', 'actual_value_millions', 'predicted_value_millions', 'value_difference_millions', 'market_value_ratio']].to_csv('overvalued_players.csv', index=False)
    
    undervalued_players = model_data.sort_values('market_value_difference').head(100)
    undervalued_players[['first_name', 'last_name', 'actual_value_millions', 'predicted_value_millions', 'value_difference_millions', 'market_value_ratio']].to_csv('undervalued_players.csv', index=False)

# Run the analysis
if __name__ == "__main__":
    try:
        analyze_actual_vs_earned_market_values()
        
        print("\nAnalysis completed successfully!")
        print("Results have been saved to CSV files and visualization plots.")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc() 