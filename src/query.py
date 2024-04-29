import os
import sqlite3
import urllib.request

# Define the URL to fetch the database from
url = 'https://techassessment.blob.core.windows.net/aiap17-assessment-data/calls.db'

# Define the local path where the database will be saved
data_folder = '../data'
db_filename = os.path.join(data_folder, 'calls.db')

# Create the data folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

# Download the database file if it doesn't exist locally
if not os.path.exists(db_filename):
    print(f'Downloading {db_filename}...')
    urllib.request.urlretrieve(url, db_filename)
    print('Download complete.')
