import os
import sqlite3
import urllib.request

url = 'https://techassessment.blob.core.windows.net/aiap17-assessment-data/calls.db'

data_folder = '../data'
db_filename = os.path.join(data_folder, 'calls.db')

os.makedirs(data_folder, exist_ok=True)

if not os.path.exists(db_filename):
    print(f'Downloading {db_filename}...')
    urllib.request.urlretrieve(url, db_filename)
    print('Download complete.')
