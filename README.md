##  Project Demo

[![Project Demo Video](https://img.youtube.com/vi/BddDLzikh78/0.jpg)](https://youtu.be/BddDLzikh78)


# QuizGen API

This is a FastAPI-based application for generating quizzes from PDF documents.

## Prerequisites

Before running the application, ensure you have the following installed:

1. Python 3.8 or higher
2. Tesseract OCR
3. Poppler (for PDF processing)
4. OpenCV

### Installing Prerequisites

1. Install Tesseract OCR:
   - Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add Tesseract to your system PATH

2. Install Poppler:
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases/
   - Add Poppler to your system PATH

3. Install OpenCV:
   - OpenCV will be installed automatically through the requirements.txt file
   - If you need to install it manually, run:
   ```bash
   pip install opencv-python
   ```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://0.0.0.0:8000`

## Testing with Postman

1. Open Postman and create a new request
2. Set the request type to POST
3. Use the endpoint: `http://0.0.0.0:8000/generate-quiz`
4. Set the request body to form-data with the following fields:
   - `file`: Select your PDF file

Example request:
- Method: POST
- URL: http://0.0.0.0:8000/generate-quiz
- Body (form-data):
  - file: [your_pdf_file.pdf]
 
