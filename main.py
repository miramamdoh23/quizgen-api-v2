# Quiz Generator API 
# Secure and optimized API for generating quiz questions from PDFs

import os
import io
import re
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# PDF & Image Processing
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import cv2
import numpy as np

# AI
import google.generativeai as genai
from google.generativeai import GenerativeModel


# 1. Application Settings

class Settings(BaseSettings):
    """Centralized application configuration"""
    
    # API Keys - Must be in .env file
    google_api_key: str
    
    # File limits
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_extensions: list = [".pdf"]
    
    # OCR settings
    ocr_dpi: int = 300  # Reduced from 400 for performance
    ocr_timeout: int = 30  # seconds
    
    # Gemini settings
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.4
    gemini_max_tokens: int = 4000
    gemini_timeout: int = 30
    
    # Question settings
    max_questions_per_page: int = 3
    min_content_length: int = 50
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Load settings
try:
    settings = Settings()
except Exception as e:
    print(f" Error loading settings: {e}")
    print(" Make sure you have a .env file with GOOGLE_API_KEY")
    raise

# 2. Data Models (Pydantic)

class QuestionOption(BaseModel):
    """Single question option"""
    text: str = Field(..., min_length=1, max_length=500)

class Question(BaseModel):
    """Question model with validation"""
    text: str = Field(..., min_length=10, max_length=1000)
    type: int = Field(..., ge=1, le=2)  # 1: Multiple choice, 2: True/False
    options: List[QuestionOption] = Field(..., min_items=2, max_items=4)
    correctAnswer: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=10)
    difficulty: str = Field(default="medium")
    page: str
    quiz_id: int
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        allowed = ['easy', 'medium', 'hard']
        if v.lower() not in allowed:
            return 'medium'
        return v.lower()
    
    @validator('options')
    def validate_options_count(cls, v, values):
        q_type = values.get('type')
        if q_type == 1 and len(v) != 4:
            raise ValueError('Multiple choice questions must have 4 options')
        if q_type == 2 and len(v) != 2:
            raise ValueError('True/False questions must have 2 options')
        return v

class QuizResponse(BaseModel):
    """API response model"""
    success: bool
    questions: List[Question]
    total_pages: int
    processed_pages: int
    total_questions: int
    message: str
    processing_time: Optional[float] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[str] = None


# 3. Logging Configuration


logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 4. Gemini AI Service

class GeminiService:
    """Secure management of Gemini AI connections"""
    
    def __init__(self):
        self._model: Optional[GenerativeModel] = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize Gemini model"""
        if self._initialized:
            return True
        
        try:
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            
            genai.configure(api_key=settings.google_api_key)
            
            generation_config = {
                "temperature": settings.gemini_temperature,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": settings.gemini_max_tokens,
                "response_mime_type": "text/plain"
            }
            
            self._model = GenerativeModel(
                settings.gemini_model,
                generation_config=generation_config
            )
            
            # Test connection
            test_response = self._model.generate_content("Test connection")
            if not test_response or not test_response.text:
                raise Exception("No response from API")
            
            self._initialized = True
            logger.info(" Gemini API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f" Failed to initialize Gemini API: {e}")
            self._initialized = False
            return False
    
    def get_model(self) -> GenerativeModel:
        """Get Gemini model instance"""
        if not self._initialized:
            if not self.initialize():
                raise HTTPException(
                    status_code=503,
                    detail="Gemini AI service is currently unavailable"
                )
        return self._model
    
    async def generate_with_timeout(
        self, 
        prompt: str, 
        timeout: int = None
    ) -> str:
        """Generate content with timeout protection"""
        timeout = timeout or settings.gemini_timeout
        
        try:
            model = self.get_model()
            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt),
                timeout=timeout
            )
            return response.text.strip()
        except asyncio.TimeoutError:
            raise HTTPException(408, "Gemini AI request timeout")
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            raise HTTPException(500, f"Question generation failed: {str(e)}")

# Create Gemini service instance
gemini_service = GeminiService()

def get_gemini_service() -> GeminiService:
    """Dependency injection for Gemini service"""
    return gemini_service

# 5. Text and Image Processing

# Pre-compile regex patterns for performance
PATTERNS = {
    'university': re.compile(
        r'THE\s+EGFPTIANE[-\s]*LEARNING\s*UNIVERSITY',
        re.IGNORECASE
    ),
    'e_learning': re.compile(r'EGFPTIANE[-\s]*LEARNING', re.IGNORECASE),
    'spaces': re.compile(r'\s+'),
    'repetition': re.compile(r'(\b\w+\b)(?:\s+\1){2,}'),
    'word_spacing': re.compile(r'([a-z])([A-Z])'),
    'extra_newlines': re.compile(r'\n\s*\n'),
}

REPLACEMENTS = {
    'university': 'THE EGYPTIAN E-LEARNING UNIVERSITY',
    'e_learning': 'EGYPTIAN E-LEARNING',
}

def clean_text(text: str) -> str:
    """Clean extracted text from OCR errors"""
    if not text or not text.strip():
        return ""
    
    # Apply patterns
    text = PATTERNS['spaces'].sub(' ', text.strip())
    text = PATTERNS['university'].sub(REPLACEMENTS['university'], text)
    text = PATTERNS['e_learning'].sub(REPLACEMENTS['e_learning'], text)
    text = PATTERNS['repetition'].sub(r'\1', text)
    text = PATTERNS['word_spacing'].sub(r'\1 \2', text)
    text = PATTERNS['extra_newlines'].sub('\n', text)
    
    return text.strip()

def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance image quality before OCR"""
    try:
        # Convert to numpy array
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(binary)
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return image


# 6. PDF Text Extraction


async def extract_text_from_pdf(
    file_bytes: bytes,
    filename: str
) -> tuple[str, int]:
    """Extract text from PDF with enhanced OCR"""
    
    try:
        text = ""
        
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"📄 Processing {total_pages} pages from {filename}")
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Direct text extraction
                    page_text = page.extract_text() or ""
                    
                    # If text is too short, try OCR
                    if len(page_text.strip()) < settings.min_content_length:
                        try:
                            # Convert only current page to image
                            images = await asyncio.wait_for(
                                asyncio.to_thread(
                                    convert_from_bytes,
                                    file_bytes,
                                    dpi=settings.ocr_dpi,
                                    first_page=page_num + 1,
                                    last_page=page_num + 1,
                                    fmt='PNG'
                                ),
                                timeout=settings.ocr_timeout
                            )
                            
                            if images:
                                img = images[0]
                                processed_img = preprocess_image(img)
                                
                                ocr_text = await asyncio.to_thread(
                                    pytesseract.image_to_string,
                                    processed_img,
                                    config='--oem 1 --psm 6 --dpi 300'
                                )
                                
                                if len(ocr_text) > len(page_text):
                                    page_text = ocr_text
                                    
                        except asyncio.TimeoutError:
                            logger.warning(f"⏱️ OCR timeout for page {page_num + 1}")
                        except Exception as ocr_error:
                            logger.warning(f"OCR error for page {page_num + 1}: {ocr_error}")
                    
                    # Clean text
                    page_text = clean_text(page_text)
                    
                    if page_text:
                        text += f"\n\n<<PAGE_{page_num + 1}>>\n{page_text}\n\n"
                    else:
                        text += f"\n\n<<PAGE_{page_num + 1}>>\nNo readable text found.\n\n"
                        
                except Exception as page_error:
                    logger.error(f"Error processing page {page_num + 1}: {page_error}")
                    continue
        
        if not text.strip():
            raise ValueError("Unable to extract any text from PDF")
        
        return text, total_pages
        
    except Exception as e:
        logger.error(f"Text extraction error: {e}")
        raise HTTPException(422, f"Failed to extract text from PDF: {str(e)}")


# 7. Question Generation

def analyze_content(content: str) -> Dict[str, Any]:
    """Analyze content quality for question generation"""
    if not content or not content.strip():
        return {
            "quality": "empty",
            "word_count": 0,
            "has_academic_content": False
        }
    
    words = content.strip().split()
    word_count = len(words)
    
    # Academic keywords
    academic_terms = [
        'management', 'organization', 'structure', 'department',
        'strategy', 'planning', 'control', 'process', 'theory',
        'analysis', 'research', 'study', 'development'
    ]
    
    has_academic = any(term in content.lower() for term in academic_terms)
    
    # Determine quality
    if word_count >= 100:
        quality = "high"
    elif word_count >= 50:
        quality = "medium"
    elif word_count >= 20:
        quality = "low"
    else:
        quality = "minimal"
    
    return {
        "quality": quality,
        "word_count": word_count,
        "has_academic_content": has_academic
    }

async def generate_questions(
    content: str,
    page_num: int,
    gemini: GeminiService
) -> List[Question]:
    """Generate questions using AI"""
    
    # Analyze content
    analysis = analyze_content(content)
    
    if analysis["quality"] == "empty":
        return []
    
    # Determine number of questions
    num_questions = {
        "high": 3,
        "medium": 2,
        "low": 1,
        "minimal": 1
    }.get(analysis["quality"], 1)
    
    # Build enhanced prompt
    prompt = f"""You are an expert in creating educational assessments.

The following content is from an academic textbook page:
{content[:1500]}

Requirements:
- Generate {num_questions} question(s) in ENGLISH ONLY
- Questions should test understanding and application, not just memorization
- Use variety in difficulty: easy, medium, hard

Question Types:
- Type 1: Multiple choice (4 options)
- Type 2: True/False (2 options)

Required Format EXACTLY:
Question: [question text]?
Type: [1 or 2]
Options: [option1, option2, option3, option4]
Answer: [correct answer]
Explanation: [detailed explanation]
Difficulty: [easy/medium/hard]

Generate exactly {num_questions} question(s) now:"""

    try:
        response = await gemini.generate_with_timeout(prompt)
        questions = parse_ai_response(response, page_num)
        
        if questions:
            logger.info(f" Generated {len(questions)} question(s) for page {page_num}")
        
        return questions
        
    except Exception as e:
        logger.error(f"Failed to generate questions for page {page_num}: {e}")
        return []

def parse_ai_response(response: str, page_num: int) -> List[Question]:
    """Parse AI response into Question objects"""
    questions = []
    
    # Split by "Question:"
    blocks = response.split("Question:")
    
    for block in blocks[1:]:
        try:
            lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            
            if len(lines) < 5:
                continue
            
            # Extract data
            question_text = lines[0].rstrip('?') + '?'
            q_data = {}
            
            for line in lines[1:]:
                if line.startswith("Type:"):
                    q_data["type"] = int(line.split(":")[1].strip())
                elif line.startswith("Options:"):
                    opts_str = line.split(":", 1)[1].strip().strip('[]')
                    opts = [o.strip().strip('"\'') for o in opts_str.split(',')]
                    q_data["options"] = [QuestionOption(text=o) for o in opts if o]
                elif line.startswith("Answer:"):
                    q_data["correctAnswer"] = line.split(":", 1)[1].strip()
                elif line.startswith("Explanation:"):
                    q_data["explanation"] = line.split(":", 1)[1].strip()
                elif line.startswith("Difficulty:"):
                    q_data["difficulty"] = line.split(":", 1)[1].strip().lower()
            
            # Validate and create
            if all(k in q_data for k in ["type", "options", "correctAnswer", "explanation"]):
                question = Question(
                    text=question_text,
                    type=q_data["type"],
                    options=q_data["options"],
                    correctAnswer=q_data["correctAnswer"],
                    explanation=q_data["explanation"],
                    difficulty=q_data.get("difficulty", "medium"),
                    page=str(page_num),
                    quiz_id=page_num
                )
                questions.append(question)
                
        except Exception as e:
            logger.warning(f"Failed to parse question block: {e}")
            continue
    
    return questions


# 8. FastAPI Application

app = FastAPI(
    title="Quiz Generator API",
    description="Generate quiz questions from PDF files using AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# 9. Startup/Shutdown Events


@app.on_event("startup")
async def startup_event():
    """Initialize on application startup"""
    logger.info(" Starting Quiz Generator API v2.0")
    
    # Initialize Gemini
    if not gemini_service.initialize():
        logger.error(" Gemini initialization failed - will retry on first request")
    
    logger.info(" Application ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info(" Shutting down Quiz Generator API")


# 10. API Endpoints

@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Quiz Generator API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Advanced PDF text extraction",
            "Enhanced OCR for images",
            "AI-powered question generation",
            "Duplicate detection",
            "Improved security"
        ],
        "endpoints": {
            "generate_quiz": "POST /generate-quiz",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health", tags=["Info"])
async def health_check():
    """System health check"""
    gemini_status = "connected" if gemini_service._initialized else "disconnected"
    
    return {
        "status": "healthy",
        "gemini_ai": gemini_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post(
    "/generate-quiz",
    response_model=QuizResponse,
    tags=["Quiz"],
    summary="Generate quiz from PDF",
    description="Upload a PDF file and receive AI-generated quiz questions",
    responses={
        200: {"description": "Quiz generated successfully"},
        413: {"description": "File too large"},
        422: {"description": "Invalid PDF file"},
        500: {"description": "Server error"}
    }
)
async def generate_quiz(
    file: UploadFile = File(..., description="PDF file to process"),
    gemini: GeminiService = Depends(get_gemini_service)
):
    """
    Generate quiz questions from uploaded PDF file
    
    Process:
    1. Extract text from PDF
    2. Apply OCR to images if needed
    3. Generate questions using AI
    4. Return structured questions
    """
    
    start_time = datetime.now()
    
    try:
        # 1. Validate file type
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type. Allowed: {settings.allowed_extensions}"
            )
        
        # 2. Read file
        file_bytes = await file.read()
        
        # 3. Validate file size
        if len(file_bytes) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f} MB"
            )
        
        logger.info(f" Uploaded file: {file.filename} ({len(file_bytes) / 1024:.1f} KB)")
        
        # 4. Extract text
        extracted_text, total_pages = await extract_text_from_pdf(
            file_bytes,
            file.filename
        )
        
        # 5. Process pages and generate questions
        all_questions = []
        processed_pages = 0
        
        page_blocks = extracted_text.split('<<PAGE_')
        
        for block in page_blocks[1:]:
            try:
                page_num = int(block.split('>>')[0])
                content = block.split('>>', 1)[1].split('\n\n')[0].strip()
                
                if content == "No readable text found.":
                    continue
                
                if len(content) < settings.min_content_length:
                    continue
                
                # Generate questions
                questions = await generate_questions(content, page_num, gemini)
                
                if questions:
                    all_questions.extend(questions)
                    processed_pages += 1
                    
            except Exception as e:
                logger.error(f"Error processing page: {e}")
                continue
        
        # 6. Remove duplicates
        unique_questions = []
        seen = set()
        
        for q in all_questions:
            key = (q.text.lower(), tuple(o.text.lower() for o in q.options))
            if key not in seen:
                seen.add(key)
                unique_questions.append(q)
        
        # 7. Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f" Completed: {len(unique_questions)} questions from {processed_pages} pages in {processing_time:.2f}s")
        
        # 8. Return response
        return QuizResponse(
            success=True,
            questions=unique_questions,
            total_pages=total_pages,
            processed_pages=processed_pages,
            total_questions=len(unique_questions),
            message=f"Successfully generated {len(unique_questions)} questions from {processed_pages} pages",
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" Error in generate_quiz: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


# 11. Main Entry Point


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )


