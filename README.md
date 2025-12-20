# QuizGen API

This is a FastAPI-based application for generating quizzes from PDF documents.

---

## Project Demo

[![Project Demo Video](https://img.youtube.com/vi/BddDLzikh78/0.jpg)](https://youtu.be/BddDLzikh78)

---

## Prerequisites

Before running the application, ensure you have the following installed:

1. Python 3.8 or higher
2. Tesseract OCR
3. Poppler (for PDF processing)
4. OpenCV

### Installing Prerequisites

**1. Install Tesseract OCR**:
- Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Add Tesseract to your system PATH

**2. Install Poppler**:
- Download from: https://github.com/oschwartz10612/poppler-windows/releases/
- Add Poppler to your system PATH

**3. Install OpenCV**:
- OpenCV will be installed automatically through the requirements.txt file
- If you need to install it manually, run:
```bash
pip install opencv-python
```

---

## Setup Instructions

**1. Create a virtual environment**:
```bash
python -m venv venv
```

**2. Activate the virtual environment**:

Windows:
```bash
.\venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

**3. Install dependencies**:
```bash
pip install -r requirements.txt
```

**4. Create a `.env` file in the root directory with your Google API key**:
```
GOOGLE_API_KEY=your_api_key_here
```

---

## Running the Application

**1. Start the FastAPI server**:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://0.0.0.0:8000`

---

## Testing with Postman

**1. Open Postman and create a new request**

**2. Set the request type to POST**

**3. Use the endpoint**: `http://0.0.0.0:8000/generate-quiz`

**4. Set the request body to form-data with the following fields**:
- `file`: Select your PDF file

### Example Request

- **Method**: POST
- **URL**: http://0.0.0.0:8000/generate-quiz
- **Body** (form-data):
  - file: [your_pdf_file.pdf]

---

## API Endpoints

### POST /generate-quiz

Generates a quiz from an uploaded PDF document.

**Request**:
- Content-Type: multipart/form-data
- Body: PDF file

**Response**:
```json
{
  "quiz": [
    {
      "question": "Sample question?",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "A"
    }
  ]
}
```

---

## Features

- PDF text extraction and processing
- OCR support for scanned documents
- Automatic quiz generation using Google AI
- RESTful API interface
- Easy integration with frontend applications

---

## Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **Tesseract OCR**: Optical character recognition
- **Poppler**: PDF rendering library
- **OpenCV**: Computer vision and image processing
- **Google Generative AI**: Quiz generation
- **Python 3.8+**: Core programming language

---

## Project Structure
```
quizgen-api/
│
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

---

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Error Handling

The API includes comprehensive error handling for:
- Invalid file formats
- Missing API keys
- PDF processing errors
- OCR failures
- Network issues

---

## Future Enhancements

- Support for additional file formats (DOCX, TXT)
- Customizable quiz difficulty levels
- Multiple choice and true/false question types
- Question bank storage and management
- User authentication and authorization
- Rate limiting and API quotas
- Batch processing for multiple documents

---

## Troubleshooting

**Issue**: Tesseract not found
- **Solution**: Ensure Tesseract is installed and added to system PATH

**Issue**: Poppler not found
- **Solution**: Ensure Poppler is installed and added to system PATH

**Issue**: API key error
- **Solution**: Verify `.env` file contains valid Google API key

**Issue**: PDF processing fails
- **Solution**: Check if PDF is corrupted or password-protected

---

## Author

**Mira Mamdoh Yousef Mossad**  
AI QA Engineer | Full Stack Developer

**Specializing in**:
- FastAPI Development
- AI/ML Integration
- API Testing & Validation
- Document Processing

**Connect**:
- Email: miramamdoh10@gmail.com
- LinkedIn: [linkedin.com/in/mira-mamdoh-a9aa78224](https://www.linkedin.com/in/mira-mamdoh-a9aa78224)
- GitHub: [github.com/miramamdoh23](https://github.com/miramamdoh23)

---

## License

This project is provided as a portfolio demonstration and educational resource.

---

## Acknowledgments

Built to demonstrate professional FastAPI development, AI integration, and document processing capabilities for quiz generation applications.

---

**Built by Mira Mamdoh**
