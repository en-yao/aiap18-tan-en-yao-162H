import sqlite3
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Disable chained assignment warning (false positive)
pd.options.mode.chained_assignment = None

def fetch_data(db_file_path: str, query: str) -> pd.DataFrame:
    """
    Fetch data from an SQLite database using the provided SQL query.

    Parameters:
    - db_file_path: Path to the SQLite database file.
    - query: SQL query to execute.

    Returns:
    - DataFrame containing the result of the query.
    """
    with sqlite3.connect(db_file_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the DataFrame.

    Parameters:
    - df: DataFrame to clean.

    Returns:
    - Cleaned DataFrame.
    """
    # Convert object columns to numeric
    numeric_columns = [
        'Daily Rainfall Total (mm)',
        'Highest 30 Min Rainfall (mm)',
        'Highest 60 Min Rainfall (mm)',
        'Highest 120 Min Rainfall (mm)',
        'Min Temperature (deg C)',
        'Maximum Temperature (deg C)',
        'Min Wind Speed (km/h)',
        'Max Wind Speed (km/h)',
        'pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central',
        'psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central'
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    # Clean and standardize categorical columns
    df['Dew Point Category'] = df['Dew Point Category'].replace(
        {
            'VH': 'Very high', 'Very High': 'Very high', 'VERY HIGH': 'Very high', 'very high': 'Very high', 'Extreme': 'Very high',
            'High': 'High', 'High Level': 'High', 'HIGH': 'High', 'H': 'High', 'high': 'High',
            'Moderate': 'Moderate', 'M': 'Moderate', 'moderate': 'Moderate', 'MODERATE': 'Moderate', 'Normal': 'Moderate',
            'Low': 'Low', 'LOW': 'Low', 'low': 'Low', 'L': 'Low', 'Below Average': 'Low',
            'Very Low': 'Very low', 'very low': 'Very low', 'VL': 'Very low', 'VERY LOW': 'Very low', 'Minimal': 'Very low'
        }
    )

    df['Wind Direction'] = df['Wind Direction'].replace(
        {
            'W': 'West', 'west': 'West', 'W.': 'West', 'WEST': 'West',
            'S': 'South', 'south': 'South', 'S.': 'South', 'SOUTH': 'South', 'Southward': 'South',
            'E': 'East', 'east': 'East', 'E.': 'East', 'EAST': 'East',
            'N': 'North', 'north': 'North', 'N.': 'North', 'NORTH': 'North', 'Northward': 'North',
            'NE': 'Northeast', 'northeast': 'Northeast', 'NE.': 'Northeast', 'NORTHEAST': 'Northeast',
            'NW': 'Northwest', 'northwest': 'Northwest', 'NW.': 'Northwest', 'NORTHWEST': 'Northwest',
            'SE': 'Southeast', 'southeast': 'Southeast', 'SE.': 'Southeast', 'SOUTHEAST': 'Southeast',
            'SW': 'Southwest', 'southwest': 'Southwest', 'SW.': 'Southwest', 'SOUTHWEST': 'Southwest'
        }
    )

    # Convert negative values to positive for 'Max Wind Speed'
    df['Max Wind Speed (km/h)'] = df['Max Wind Speed (km/h)'].abs()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Impute missing values with the median
    columns_with_missing_values = numeric_columns + ['Sunshine Duration (hrs)', 'Cloud Cover (%)']
    df[columns_with_missing_values] = df[columns_with_missing_values].fillna(df[columns_with_missing_values].median())

    # Bin numeric features
    df['Daily Rainfall'] = pd.cut(
        df['Daily Rainfall Total (mm)'],
        bins=[-float('inf'), 0, 25, 50, 100, 200, float('inf')],
        labels=['No rain', 'Very light', 'Light', 'Moderate', 'Heavy', 'Extreme'],
        include_lowest=True
    )

    df['Min Temperature'] = pd.cut(
        df['Min Temperature (deg C)'],
        bins=[-float('inf'), 15, 25, 35, float('inf')],
        labels=['Cold', 'Mild', 'Hot', 'Very hot'],
        right=False
    )

    df['Maximum Temperature'] = pd.cut(
        df['Maximum Temperature (deg C)'],
        bins=[-float('inf'), 15, 25, 35, float('inf')],
        labels=['Cold', 'Mild', 'Hot', 'Very hot'],
        right=False
    )

    df['Relative Humidity'] = pd.cut(
        df['Relative Humidity (%)'],
        bins=[-float('inf'), 30, 60, 90, float('inf')],
        labels=['Low', 'Moderate', 'High', 'Very High'],
        right=False
    )

    pm25_bins = [-float('inf'), 12, 35, 55, 150, 250, float('inf')]
    pm25_labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    df['PM25 North'] = pd.cut(df['pm25_north'], bins=pm25_bins, labels=pm25_labels)
    df['PM25 South'] = pd.cut(df['pm25_south'], bins=pm25_bins, labels=pm25_labels)
    df['PM25 East'] = pd.cut(df['pm25_east'], bins=pm25_bins, labels=pm25_labels)
    df['PM25 West'] = pd.cut(df['pm25_west'], bins=pm25_bins, labels=pm25_labels)
    df['PM25 Central'] = pd.cut(df['pm25_central'], bins=pm25_bins, labels=pm25_labels)

    psi_bins = [-float('inf'), 50, 100, 150, 200, 300, float('inf')]
    psi_labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    df['PSI North'] = pd.cut(df['psi_north'], bins=psi_bins, labels=psi_labels)
    df['PSI South'] = pd.cut(df['psi_south'], bins=psi_bins, labels=psi_labels)
    df['PSI East'] = pd.cut(df['psi_east'], bins=psi_bins, labels=psi_labels)
    df['PSI West'] = pd.cut(df['psi_west'], bins=psi_bins, labels=psi_labels)
    df['PSI Central'] = pd.cut(df['psi_central'], bins=psi_bins, labels=psi_labels)

    # Feature engineering
    df['temp_range'] = df['Maximum Temperature (deg C)'] - df['Min Temperature (deg C)']
    df['Temperature Range'] = pd.cut(
        df['temp_range'],
        bins=[-float('inf'), 5, 10, float('inf')],
        labels=['Low', 'Moderate', 'High']
    )

    df['wind_speed_range'] = df['Max Wind Speed (km/h)'] - df['Min Wind Speed (km/h)']
    df['Wind Speed Range'] = pd.cut(
        df['wind_speed_range'],
        bins=[-float('inf'), 10, 30, 40, float('inf')],
        labels=['Low', 'Moderate', 'High', 'Extreme']
    )

    # Convert target feature to numeric
    label_encoder = preprocessing.LabelEncoder()
    df['Daily Solar Panel Efficiency']= label_encoder.fit_transform(df['Daily Solar Panel Efficiency'])

    # Remove outliers for "Wet Bulb Temperature"
    df = df[df['Wet Bulb Temperature (deg F)'].between(0, 100)]

    return df