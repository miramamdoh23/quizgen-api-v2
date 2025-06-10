import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pdfplumber
from google.generativeai import GenerativeModel, configure
import logging
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
import random
import cv2
import numpy as np
import difflib
import re
import json
from typing import List, Dict, Any

# Create a FastAPI application to provide an API interface
app = FastAPI()

# Configure logging to track errors and information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API key from environment variables
configure(api_key=os.getenv("GEMINI_API_KEY"))
model = GenerativeModel("gemini-2.0-flash")

# Enhanced function to clean extracted text from OCR errors
def clean_extracted_text(text):
    """
    Enhanced text cleaning with comprehensive OCR error corrections.
    """
    if not text or not text.strip():
        return text
    
    # Normalize spaces first
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Comprehensive OCR corrections dictionary
    corrections = {
        # University name corrections
        r'THE\s+EGFPTIANE[-\s]*LEARNING\s*UNIVERSITY': 'THE EGYPTIAN E-LEARNING UNIVERSITY',
        r'EGFPTIANE[-\s]*LEARNING': 'EGYPTIAN E-LEARNING',
        r'E[-\s]*LEARNING\s*UNIVERSITY': 'E-LEARNING UNIVERSITY',
        
        # Common OCR mistakes
        r'[wW][aA][tT][cC][uU][iI][nN][eE]': 'Watching',
        r'THANK\s*["\']?\s*A\s*YOU': 'THANK YOU',
        r'ALWAYS\s+LEARNINC': 'ALWAYS LEARNING',
        r'pudiseconomies': 'diseconomies',
        r'MANAGER(?:\s+MANAGER)+': 'MANAGER',  # Remove repeated MANAGER
        r'Manager(?:\s+Manager)+': 'Manager',   # Remove repeated Manager
        
        # Academic terms
        r'PEARSON': 'Pearson',
        r'Departmentalization': 'Departmentalization',
        r'Organizational': 'Organizational',
        r'Management': 'Management',
        
        # Common formatting issues
        r'(\w)\s*-\s*(\w)': r'\1-\2',  # Fix hyphenated words
        r'([a-z])([A-Z])': r'\1 \2',   # Add space between camelCase
        r'\b(\d+)\s*\.\s*(\d+)\b': r'\1.\2',  # Fix decimal numbers
        
        # Remove excessive repetition
        r'(\b\w+\b)(?:\s+\1){2,}': r'\1',  # Remove word repeated 3+ times
    }
    
    # Apply corrections
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Clean up extra spaces again after corrections
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

# Enhanced function to preprocess images before OCR
def preprocess_image_for_ocr(image):
    """
    Advanced image preprocessing for better OCR accuracy.
    """
    try:
        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple enhancement techniques
        # 1. CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Noise reduction
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # 3. Sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 4. Adaptive threshold for better text separation
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 5. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        processed_image = Image.fromarray(cleaned)
        return processed_image
        
    except Exception as e:
        logger.error(f"Error in advanced image preprocessing: {e}")
        return image

# Enhanced text extraction with better error handling
async def extract_text_from_pdf(file):
    """
    Enhanced PDF text extraction with improved OCR and error handling.
    """
    try:
        file_extension = os.path.splitext(file.filename.lower())[1]
        if file_extension != '.pdf':
            raise ValueError(f"Uploaded file is not a PDF. Detected extension: {file_extension}")

        await file.seek(0)
        file_bytes = await file.read()
        text = ""
        
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            
            # Convert all pages to images once for OCR
            try:
                images = convert_from_bytes(file_bytes, dpi=400, fmt='PNG')
            except Exception as e:
                logger.warning(f"PDF to image conversion failed: {e}")
                images = []
            
            for page_num, page in enumerate(pdf.pages):
                logger.info(f"Processing page {page_num + 1}/{total_pages}")
                
                # Extract text using pdfplumber
                page_text = ""
                try:
                    page_text = page.extract_text() or ""
                    if page_text:
                        page_text = clean_extracted_text(page_text)
                        logger.info(f"Direct text extraction from page {page_num + 1}: {len(page_text)} chars")
                except Exception as e:
                    logger.warning(f"Direct text extraction failed for page {page_num + 1}: {e}")
                
                # OCR extraction
                ocr_text = ""
                if page_num < len(images):
                    try:
                        img = images[page_num]
                        processed_img = preprocess_image_for_ocr(img)
                        
                        # Multiple OCR configurations for better results
                        ocr_configs = [
                            '--oem 1 --psm 6 -c preserve_interword_spaces=1 --dpi 400',
                            '--oem 1 --psm 4 -c preserve_interword_spaces=1 --dpi 400',
                            '--oem 1 --psm 3 -c preserve_interword_spaces=1 --dpi 400'
                        ]
                        
                        best_ocr = ""
                        for config in ocr_configs:
                            try:
                                current_ocr = pytesseract.image_to_string(processed_img, config=config)
                                if len(current_ocr) > len(best_ocr):
                                    best_ocr = current_ocr
                            except:
                                continue
                        
                        if best_ocr:
                            ocr_text = clean_extracted_text(best_ocr)
                            logger.info(f"OCR text from page {page_num + 1}: {len(ocr_text)} chars")
                        
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for page {page_num + 1}: {ocr_error}")
                
                # Combine and choose the best text
                final_text = ""
                if page_text and ocr_text:
                    # Use similarity to determine which text is better
                    if len(ocr_text) > len(page_text) * 1.5:  # OCR significantly longer
                        final_text = ocr_text
                    elif len(page_text) > 50:  # Direct extraction has substantial content
                        final_text = page_text
                    else:
                        # Combine both
                        final_text = f"{page_text} {ocr_text}".strip()
                elif page_text:
                    final_text = page_text
                elif ocr_text:
                    final_text = ocr_text
                
                # Final cleaning
                if final_text:
                    final_text = clean_extracted_text(final_text)
                    # Remove excessive whitespace and normalize
                    final_text = re.sub(r'\n\s*\n', '\n', final_text)
                    final_text = re.sub(r'[ \t]+', ' ', final_text)
                
                # Add to main text with page markers
                if final_text.strip():
                    text += f"<PAGE{page_num + 1}>\n<CONTENT>\n{final_text.strip()}\n</CONTENT>\n</PAGE{page_num + 1}>\n"
                else:
                    text += f"<PAGE{page_num + 1}>\n<CONTENT>\nNo readable text found on this page.\n</CONTENT>\n</PAGE{page_num + 1}>\n"

        if not text.strip():
            raise ValueError("Unable to extract any text from the PDF.")
            
        return text, total_pages
        
    except Exception as e:
        logger.error(f"Error in extracting text from PDF: {e}")
        raise

# Enhanced content analysis for better question generation
def analyze_content_quality(content: str) -> Dict[str, Any]:
    """
    Analyze content quality to determine appropriate question generation strategy.
    """
    if not content or not content.strip():
        return {"quality": "empty", "word_count": 0, "key_terms": [], "has_numbers": False}
    
    words = content.strip().split()
    word_count = len(words)
    
    # Extract key terms (meaningful words)
    key_terms = []
    numbers = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if len(clean_word) > 3 and clean_word.isalpha():
            key_terms.append(clean_word)
        elif clean_word.isdigit() and 1 <= len(clean_word) <= 4:
            numbers.append(clean_word)
    
    # Remove duplicates while preserving order
    key_terms = list(dict.fromkeys(key_terms))
    numbers = list(dict.fromkeys(numbers))
    
    # Determine content quality
    quality = "empty"
    if word_count >= 100:
        quality = "high"
    elif word_count >= 50:
        quality = "medium"
    elif word_count >= 20:
        quality = "low"
    elif word_count >= 5:
        quality = "minimal"
    
    return {
        "quality": quality,
        "word_count": word_count,
        "key_terms": key_terms[:10],  # Top 10 key terms
        "numbers": numbers[:5],       # Top 5 numbers
        "has_academic_content": any(term in content.lower() for term in 
                                  ['management', 'organization', 'structure', 'department', 
                                   'strategy', 'planning', 'control', 'process'])
    }

# Enhanced AI question generation with better prompts
def generate_ai_questions(content: str, page_num: int, content_analysis: Dict) -> List[Dict]:
    """
    Generate questions using AI with enhanced prompts based on content analysis.
    """
    try:
        # Determine number of questions based on content quality
        num_questions = {
            "high": 3,
            "medium": 2,
            "low": 1,
            "minimal": 1,
            "empty": 0
        }.get(content_analysis["quality"], 1)
        
        if num_questions == 0:
            return []
        
        # Create context-aware prompt
        prompt = f"""
You are an educational assessment expert. Generate {num_questions} high-quality quiz questions based on the following academic content.

CONTENT:
{content}

GUIDELINES:
1. Generate questions in ENGLISH ONLY
2. Create questions at different difficulty levels (easy, medium, hard)
3. Focus on understanding, application, and analysis - not just memorization
4. Avoid generic questions like "What is mentioned on this page?"
5. Make questions specific to the actual content provided
6. Use variety: multiple choice (Type 1) and true/false (Type 2)

QUESTION TYPES:
- Type 1: Multiple choice with 4 options
- Type 2: True/False

FORMAT REQUIREMENTS:
For each question, use this EXACT format:
Question: [question text]?
Type: [1 or 2]
Options: [option1, option2, option3, option4] (for Type 1) OR [True, False] (for Type 2)
Answer: [correct answer]
Explanation: [why this answer is correct]
Difficulty: [easy/medium/hard]
Page: {page_num}

QUALITY STANDARDS:
- Questions should test comprehension and critical thinking
- Avoid questions that can be answered without reading the content
- Make distractors plausible but clearly incorrect
- Ensure explanations add educational value

Generate exactly {num_questions} question(s):
"""

        generation_config = {
            "temperature": 0.4,  # Lower temperature for more consistent formatting
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2000,
            "response_mime_type": "text/plain"
        }

        response = model.generate_content(prompt, generation_config=generation_config)
        raw_response = response.text.strip()
        
        questions = parse_enhanced_ai_response(raw_response, page_num)
        
        if questions:
            logger.info(f"Generated {len(questions)} AI questions for page {page_num}")
            return questions
        else:
            logger.warning(f"AI question parsing failed for page {page_num}")
            return []
        
    except Exception as e:
        logger.error(f"AI question generation failed for page {page_num}: {e}")
        return []

def parse_enhanced_ai_response(raw_response: str, page_num: int) -> List[Dict]:
    """
    Parse AI response with enhanced error handling and validation.
    """
    questions = []
    
    # Split by "Question:" to get individual questions
    question_blocks = raw_response.split("Question:")
    
    for block in question_blocks[1:]:  # Skip first empty block
        try:
            lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            if len(lines) < 5:
                continue
            
            # Parse components
            question_text = lines[0].rstrip('?') + '?'
            
            question_data = {}
            for line in lines[1:]:
                if line.startswith("Type:"):
                    question_data["type"] = int(line.split(":")[1].strip())
                elif line.startswith("Options:"):
                    options_str = line.split(":", 1)[1].strip()
                    # Parse options - remove brackets and split by comma
                    options_str = options_str.strip('[]')
                    options = [opt.strip().strip('"\'') for opt in options_str.split(',')]
                    question_data["options"] = [{"text": opt} for opt in options if opt]
                elif line.startswith("Answer:"):
                    question_data["correctAnswer"] = line.split(":", 1)[1].strip()
                elif line.startswith("Explanation:"):
                    question_data["explanation"] = line.split(":", 1)[1].strip()
                elif line.startswith("Difficulty:"):
                    question_data["difficulty"] = line.split(":", 1)[1].strip().lower()
            
            # Validate question data
            if (question_text and 
                question_data.get("type") in [1, 2] and 
                question_data.get("options") and 
                question_data.get("correctAnswer") and 
                question_data.get("explanation")):
                
                question = {
                    "text": question_text,
                    "type": question_data["type"],
                    "options": question_data["options"],
                    "correctAnswer": question_data["correctAnswer"],
                    "explanation": question_data["explanation"],
                    "difficulty": question_data.get("difficulty", "medium"),
                    "page": str(page_num),
                    "quiz_id": page_num
                }
                
                # Additional validation
                if question_data["type"] == 1 and len(question_data["options"]) == 4:
                    questions.append(question)
                elif question_data["type"] == 2 and len(question_data["options"]) == 2:
                    questions.append(question)
                
        except Exception as e:
            logger.warning(f"Failed to parse question block: {e}")
            continue
    
    return questions

# Enhanced fallback question generation
def generate_content_based_questions(content: str, page_num: int, content_analysis: Dict) -> List[Dict]:
    """
    Generate questions based on content analysis when AI fails.
    """
    questions = []
    
    if content_analysis["has_academic_content"]:
        # Generate academic-focused questions
        academic_terms = ["management", "organization", "structure", "department", 
                         "strategy", "planning", "control", "process"]
        
        found_terms = [term for term in academic_terms if term in content.lower()]
        
        if found_terms:
            term = found_terms[0]
            question = {
                "text": f"Which concept is discussed in relation to {term} in this content?",
                "type": 1,
                "options": [
                    {"text": term.capitalize()},
                    {"text": "Financial Analysis"},
                    {"text": "Market Research"},
                    {"text": "Product Development"}
                ],
                "correctAnswer": term.capitalize(),
                "explanation": f"The content discusses concepts related to {term}",
                "difficulty": "medium",
                "page": str(page_num),
                "quiz_id": page_num
            }
            questions.append(question)
    
    # Generate questions from key terms
    if content_analysis["key_terms"]:
        key_term = content_analysis["key_terms"][0]
        question = {
            "text": f"What key term is prominently featured in this content?",
            "type": 1,
            "options": [
                {"text": key_term.capitalize()},
                {"text": "Innovation"},
                {"text": "Technology"},
                {"text": "Competition"}
            ],
            "correctAnswer": key_term.capitalize(),
            "explanation": f"The term '{key_term}' appears prominently in the content",
            "difficulty": "easy",
            "page": str(page_num),
            "quiz_id": page_num
        }
        questions.append(question)
    
    return questions

# Enhanced main question generation function
def generate_questions_from_content(content: str, page_num: int) -> List[Dict]:
    """
    Main function to generate questions with enhanced logic and fallbacks.
    """
    try:
        # Analyze content quality
        content_analysis = analyze_content_quality(content)
        logger.info(f"Page {page_num} analysis: {content_analysis}")
        
        # Skip empty or very poor content
        if content_analysis["quality"] == "empty":
            return []
        
        # Try AI generation first for good quality content
        if content_analysis["quality"] in ["high", "medium"]:
            questions = generate_ai_questions(content, page_num, content_analysis)
            if questions:
                return questions
        
        # Fallback to content-based questions
        return generate_content_based_questions(content, page_num, content_analysis)
        
    except Exception as e:
        logger.error(f"Error in generate_questions_from_content for page {page_num}: {e}")
        return []

# Enhanced duplicate detection
def is_duplicate_content(content1: str, content2: str, threshold: float = 0.85) -> bool:
    """
    Enhanced duplicate detection with better similarity measures.
    """
    if not content1 or not content2:
        return False
    
    if len(content1) < 20 or len(content2) < 20:
        return False
    
    # Multiple similarity measures
    sequence_similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
    
    # Word-based similarity
    words1 = set(content1.lower().split())
    words2 = set(content2.lower().split())
    
    if not words1 or not words2:
        return False
    
    word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
    
    # Combined similarity score
    combined_similarity = (sequence_similarity + word_similarity) / 2
    
    return combined_similarity > threshold

# Main endpoint with enhanced processing
@app.post("/generate-quiz")
async def generate_quiz(file: UploadFile = File(...)):
    """
    Enhanced endpoint with better processing and error handling.
    """
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Extract text with enhanced processing
        extracted_text, total_pages = await extract_text_from_pdf(file)
        
        all_questions = []
        pages_content = {}
        processed_pages = set()
        
        # Parse pages
        page_blocks = extracted_text.split('</PAGE')
        
        for block in page_blocks:
            if '<PAGE' not in block:
                continue
            
            try:
                page_num = int(block.split('<PAGE')[1].split('>')[0])
                content = block.split('<CONTENT>\n')[1].split('\n</CONTENT>')[0].strip()
                
                # Skip if no meaningful content
                if content == "No readable text found on this page.":
                    continue
                
                # Check for duplicates
                is_duplicate = False
                for existing_page, existing_content in pages_content.items():
                    if is_duplicate_content(content, existing_content):
                        logger.info(f"Skipping duplicate page {page_num}, similar to page {existing_page}")
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                pages_content[page_num] = content
                processed_pages.add(page_num)
                
                # Generate questions
                questions = generate_questions_from_content(content, page_num)
                if questions:
                    all_questions.extend(questions)
                    logger.info(f"Generated {len(questions)} questions for page {page_num}")
                else:
                    logger.info(f"No questions generated for page {page_num}")
            
            except Exception as e:
                logger.error(f"Error processing page block: {e}")
                continue
        
        # Remove duplicate questions
        unique_questions = []
        seen_questions = set()
        
        for q in all_questions:
            question_key = (q["text"].lower(), tuple(opt["text"].lower() for opt in q["options"]))
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                unique_questions.append(q)
            else:
                logger.info(f"Removed duplicate question: {q['text'][:50]}...")
        
        logger.info(f"Final results: {len(unique_questions)} unique questions from {len(processed_pages)} pages")
        
        return JSONResponse({
            "questions": unique_questions,
            "total_pages": total_pages,
            "processed_pages": len(processed_pages),
            "total_questions": len(unique_questions),
            "message": f"Successfully generated {len(unique_questions)} questions from {len(processed_pages)} pages"
        })
        
    except Exception as e:
        logger.error(f"Error in generate_quiz: {e}")
        return JSONResponse({
            "error": str(e),
            "questions": [],
            "message": "Failed to generate questions"
        }, status_code=500)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Enhanced Quiz Generator API v2.0",
        "features": [
            "Advanced OCR text extraction",
            "Comprehensive text cleaning",
            "AI-powered question generation",
            "Duplicate detection",
            "Content quality analysis"
        ],
        "usage": "POST /generate-quiz with PDF file"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)