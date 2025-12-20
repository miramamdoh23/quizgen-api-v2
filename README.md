# Quiz Generator API

[![Project Demo Video](https://img.youtube.com/vi/BddDLzikh78/0.jpg)](https://youtu.be/BddDLzikh78)

An intelligent API that generates quiz questions from PDF files using AI (Google Gemini). Perfect for educators, students, and e-learning platforms.

---

## Features

- **AI-Powered**: Uses Google Gemini 2.0 Flash for intelligent question generation
- **PDF Processing**: Extracts text from PDFs with advanced OCR support
- **Image Enhancement**: Preprocesses images for better text recognition
- **Smart Questions**: Generates multiple-choice and true/false questions
- **Content Analysis**: Analyzes content quality to determine optimal question count
- **High Performance**: Async processing with timeout protection
- **Secure**: Input validation, file size limits, and error handling
- **Quality Control**: Duplicate detection and content validation

---

## Technologies Used

- **FastAPI**: Modern Python web framework
- **Google Gemini AI**: Advanced language model for question generation
- **pdfplumber**: PDF text extraction
- **pytesseract**: OCR for image-based text
- **OpenCV**: Image preprocessing
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

---

## Requirements

- Python 3.8+
- Google Gemini API Key
- Tesseract OCR installed on system

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/miramamdoh23/quiz-generator-api.git
cd quiz-generator-api
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### 4. Configure Environment Variables

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
ENVIRONMENT=development
DEBUG=false
MAX_FILE_SIZE=10485760
OCR_DPI=300
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.4
```

### 5. Run the Application
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

---

## API Documentation

### Endpoints

#### GET /
Get API information and available endpoints.

#### GET /health
Check system health and Gemini AI connection status.

#### POST /generate-quiz
Generate quiz questions from a PDF file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: PDF file (max 10MB)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/generate-quiz" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/generate-quiz"
files = {"file": open("your_document.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "questions": [
    {
      "text": "What is the primary function of management?",
      "type": 1,
      "options": [
        {"text": "Planning and organizing resources"},
        {"text": "Marketing products"},
        {"text": "Accounting"},
        {"text": "Legal compliance"}
      ],
      "correctAnswer": "Planning and organizing resources",
      "explanation": "Management primarily involves planning, organizing, leading, and controlling organizational resources.",
      "difficulty": "medium",
      "page": "1",
      "quiz_id": 1
    }
  ],
  "total_pages": 5,
  "processed_pages": 5,
  "total_questions": 12,
  "message": "Successfully generated 12 questions from 5 pages",
  "processing_time": 8.45
}
```

### Interactive API Documentation

Visit these URLs when the server is running:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Project Structure
```
quiz-generator-api/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── .gitignore          # Git ignore file
├── README.md           # This file
└── tests/              # Test files (optional)
```

---

## Configuration

Edit settings in `.env` or modify the `Settings` class in `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| MAX_FILE_SIZE | 10MB | Maximum PDF file size |
| OCR_DPI | 300 | OCR resolution |
| GEMINI_MODEL | gemini-2.0-flash | AI model version |
| GEMINI_TEMPERATURE | 0.4 | AI creativity (0-1) |
| MAX_QUESTIONS_PER_PAGE | 3 | Max questions per page |

---

## Testing

Test the API using the interactive documentation at `/docs` or use these example commands:
```bash
# Health check
curl http://localhost:8000/health

# Generate quiz
curl -X POST "http://localhost:8000/generate-quiz" \
  -F "file=@test.pdf"
```

---

## Troubleshooting

### Common Issues

**1. "GOOGLE_API_KEY not found"**
- Ensure `.env` file exists with valid API key
- Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

**2. "Tesseract not found"**
- Install Tesseract OCR on your system
- Verify installation: `tesseract --version`

**3. "Failed to extract text from PDF"**
- Check if PDF is password-protected
- Ensure PDF is not corrupted
- Try with a different PDF file

**4. "Gemini AI service unavailable"**
- Check your internet connection
- Verify API key is valid
- Check Gemini API status

---

## Deployment

### Deploy to Production

1. Set `ENVIRONMENT=production` in `.env`
2. Update CORS origins in `main.py`
3. Use production-grade server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y tesseract-ocr

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t quiz-generator-api .
docker run -p 8000:8000 --env-file .env quiz-generator-api
```

---

## Key Features Demonstrated

### Technical Implementation
- FastAPI framework for high-performance REST API
- Google Gemini AI integration for intelligent question generation
- Advanced PDF text extraction with OCR fallback
- Image preprocessing for improved text recognition
- Async processing with timeout protection
- Comprehensive error handling and validation

### Quality Assurance
- Input validation and sanitization
- File size limits and type checking
- Duplicate question detection
- Content quality analysis
- Response time optimization

### Production-Ready Features
- Environment-based configuration
- Health check endpoints
- Interactive API documentation
- CORS support for web applications
- Structured logging and error reporting

---

## Skills Demonstrated

- **API Development**: RESTful API design with FastAPI
- **AI Integration**: Google Gemini API implementation
- **Document Processing**: PDF extraction and OCR
- **Image Processing**: OpenCV for preprocessing
- **Async Programming**: Efficient async/await patterns
- **Error Handling**: Comprehensive exception management
- **Testing**: API endpoint testing and validation
- **Documentation**: Clear API documentation with examples
- **Deployment**: Production deployment strategies

---

## Use Cases

This API is ideal for:

- **Educational Platforms**: Automated quiz generation for e-learning
- **Content Creators**: Quick quiz creation from study materials
- **Training Programs**: Assessment generation for corporate training
- **Academic Institutions**: Automated test creation for educators
- **Self-Study Tools**: Practice question generation for students

---

## Future Enhancements

- Support for additional file formats (DOCX, TXT, PPTX)
- Custom question difficulty levels
- Question type customization (essay, fill-in-blank)
- Batch processing for multiple documents
- Question bank management
- User authentication and authorization
- Analytics and usage tracking
- Multi-language support
- Question quality scoring

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

Built to demonstrate professional FastAPI development, AI integration, and document processing capabilities for intelligent quiz generation applications.

---

**Built by Mira Mamdoh**
