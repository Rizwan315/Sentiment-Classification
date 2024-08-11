import zipfile
import os

# Extract the dataset
zip_path = 'dataset/CustomerReviewData.zip'
extracted_folder = 'dataset/CustomerReviewData'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# List the contents of the extracted folder
extracted_files = os.listdir(extracted_folder)
print(f"Extracted files: {extracted_files}")
