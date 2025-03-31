from azure.storage.blob import ContainerClient
import os

# The SAS URL for the Stanford dataset
SAS_URL = "your url"

# Specify your local download folder
DOWNLOAD_FOLDER = "C:/Users/zacha/mlCancerData"

# Create the container client
container_client = ContainerClient.from_container_url(SAS_URL)

# Ensure the local download directory exists
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Download all blobs from the container
print("Starting download of blobs...")
for blob in container_client.list_blobs():
    blob_name = blob.name  # Get the blob's name
    download_path = os.path.join(DOWNLOAD_FOLDER, blob_name)  # Local file path

    # Create subdirectories if needed
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    # Download blob content
    print(f"Downloading {blob_name}...")
    with open(download_path, "wb") as file:
        downloader = container_client.download_blob(blob.name)
        file.write(downloader.readall())

print("All blobs downloaded successfully!")
