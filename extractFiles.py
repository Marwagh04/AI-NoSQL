import os
import zipfile
from tkinter import Tk, filedialog

# Define the directory structure
base_dir = os.path.join(os.getcwd(), "data")
uploads_dir = os.path.join(base_dir, "uploads")
processed_dir = os.path.join(base_dir, "processed")

def setup_directories():
    """
    Create the necessary directory structure if it doesn't exist.
    """
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Directories created:\nUploads: {uploads_dir}\nProcessed: {processed_dir}")

def upload_files():
    """
    Use a file dialog to select and upload ZIP files to the uploads directory.
    """
    print("Please select your ZIP files:")
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_paths = filedialog.askopenfilenames(title="Select ZIP files", filetypes=[("ZIP files", "*.zip")])
    
    if not file_paths:
        print("No files selected.")
        return

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        destination = os.path.join(uploads_dir, file_name)
        try:
            # Copy the file to the uploads directory
            with open(file_path, "rb") as src_file:
                with open(destination, "wb") as dest_file:
                    dest_file.write(src_file.read())
            print(f"File uploaded to: {destination}")
        except Exception as e:
            print(f"Failed to upload {file_name}: {e}")

def extract_zip_files():
    """
    Extract all ZIP files from the uploads directory to the processed directory.
    """
    for file_name in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, file_name)
        if zipfile.is_zipfile(file_path):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    extract_path = os.path.join(processed_dir, os.path.splitext(file_name)[0])
                    os.makedirs(extract_path, exist_ok=True)
                    zip_ref.extractall(extract_path)
                    print(f"Extracted {file_name} to {extract_path}")
            except Exception as e:
                print(f"Failed to extract {file_name}: {e}")
        else:
            print(f"Skipping non-ZIP file: {file_name}")

# Main script
if __name__ == "__main__":
    setup_directories()
    
    # Step 1: Upload ZIP files
    upload_files()
    
    # Step 2: Extract uploaded ZIP files
    extract_zip_files()
