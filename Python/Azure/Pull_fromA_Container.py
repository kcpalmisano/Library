
import pandas as pd
from io import StringIO 
from azure.storage.blob import BlobServiceClient

# Replace with your Azure Storage account details
connection_string = "YOUR_KEY_GOES_HERE"   ###Security + networking -> Access keys -> connection strings
container_name = "aipoc/test"
blob_name = "filename.csv"

# Create a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get a BlobClient to interact with the blob
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# Download the blob data as a string (for CSV)
blob_data = blob_client.download_blob().readall()

# Convert the blob data to a pandas DataFrame
# For CSV data
df = pd.read_csv(StringIO(blob_data.decode('utf-8')))

# For Excel data, use BytesIO
# from io import BytesIO
# df = pd.read_excel(BytesIO(blob_data))
