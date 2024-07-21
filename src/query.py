import os
import urllib.request

# Define the database URLs and filenames
URLS = {
    'weather': 'https://techassessment.blob.core.windows.net/aiap18-assessment-data/weather.db',
    'air_quality': 'https://techassessment.blob.core.windows.net/aiap18-assessment-data/air_quality.db'
}
DATABASES = {
    'weather': 'weather.db',
    'air_quality': 'air_quality.db'
}

def download_data(url, filename):
    """
    Download the database file from the given URL and save it to the specified filename.
    Creates the necessary directories if they don't exist.
    """
    data_folder = '../data'
    file_path = os.path.join(data_folder, filename)

    # Create the data folder if it does not exist
    os.makedirs(data_folder, exist_ok=True)

    # Download the database file if it does not exist
    if not os.path.exists(file_path):
        print(f'Downloading {file_path}...')
        urllib.request.urlretrieve(url, file_path)
        print('Download complete.')
    else:
        print(f'{file_path} already exists.')

def main():
    """
    Main function that handles downloading of all datasets.
    """
    for key, url in URLS.items():
        download_data(url, DATABASES[key])

if __name__ == "__main__":
    main()
