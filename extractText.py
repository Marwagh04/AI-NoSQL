import os
from PyPDF2 import PdfReader

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

# Function to process PDFs in the specified directory
def process_pdf_files(input_dir, output_dir):
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Iterate through PDF files in the input directory
        for file in os.listdir(input_dir):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(input_dir, file)
                # Extract text from PDF
                text = extract_text_from_pdf(pdf_path)
                if text:
                    # Save extracted text to .txt file
                    txt_filename = os.path.splitext(file)[0] + ".txt"
                    txt_file_path = os.path.join(output_dir, txt_filename)
                    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                        txt_file.write(text)
                    print(f"Text extracted and saved to {txt_file_path}")
    except Exception as e:
        print(f"Error processing PDF files: {e}")

if __name__ == "__main__":
    # Input directory containing PDF files
    input_directory = "data/processed/data"  # Adjust the path as needed
    # Output directory for extracted text files
    output_directory = "data/processed/texts"  # Path to save extracted text files

    # Process PDF files
    process_pdf_files(input_directory, output_directory)
