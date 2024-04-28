import sqlite3
import pandas as pd

def fetch_data(db_file_path, query) -> pd.DataFrame:
    conn = sqlite3.connect(db_file_path)
    df = pd.read_sql_query(query, conn)
    return df


def clean(df) -> pd.DataFrame:
    # Remove irrelevant columns
    df.drop(['Device Battery', 'Timestamp'], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values in 'Financial Loss' with mean
    df.loc[:, 'Financial Loss'] = df.loc[:, 'Financial Loss'].fillna(df.loc[:, 'Financial Loss'].mean())

    # Keep rows with positive values in 'Call Duration' and 'Financial Loss'
    df = df.loc[(df['Call Duration'] >= 0) & (df['Financial Loss'] >= 0), :]

    # Replace 'MM' with '95' in 'Country Prefix'
    df['Country Prefix'] = df['Country Prefix'].replace('MM', '95')

    # Replace 'Whats App' with 'WhatsApp' in 'Call Type'
    df['Call Type'] = df['Call Type'].replace('Whats App', 'WhatsApp')

    return df
