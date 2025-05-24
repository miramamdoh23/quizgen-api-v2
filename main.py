from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import PyPDF2
import google.generativeai as genai
from typing import List, Dict
import os
from dotenv import load_dotenv
import time
import requests
import json
import re

# تحميل ملف .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
print("Loaded .env file successfully")
print(f"GEMINI_API_KEY from env: {os.getenv('GEMINI_API_KEY')}")

app = FastAPI()

# إعداد Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    raise ValueError(f"Failed to configure Gemini API. Invalid or unauthorized API key: {str(e)}")

# إعداد الـ .NET Endpoint (لسه مستنيين الـ URL من الـ .NET team)
DOTNET_ENDPOINT_URL = "https://api-dotnet-team.example.com/v1/save-questions"  # غيّري الـ URL ده باللي هيدوهولك
DOTNET_AUTH_TOKEN = "Bearer your-auth-token-here"  # غيّري الـ token ده لو فيه

# دالة لإرسال الأسئلة للـ .NET Endpoint
def send_questions_to_dotnet(questions: List[Dict]) -> Dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": DOTNET_AUTH_TOKEN  # لو مفيش token، شيلي السطر ده
    }
    
    try:
        response = requests.post(DOTNET_ENDPOINT_URL, headers=headers, json=questions)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        raise HTTPException(status_code=response.status_code, detail=f"Error sending questions to .NET endpoint: {response.text}")
    except requests.exceptions.RequestException as req_err:
        raise HTTPException(status_code=500, detail=f"Failed to connect to .NET endpoint: {str(req_err)}")

# دالة لتنظيف النصوص
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    if text.count('(') > text.count(')'):
        text += ')'
    elif text.count(')') > text.count('('):
        text = '(' + text
    text = re.sub(r'[^\w\s.,()-]', '', text)
    return text

# دالة لتوليد أسئلة مع التعامل مع النصوص القصيرة
def generate_questions_from_pdf(pdf_text: str, page_number: int, difficulty_level: str = None, retries: int = 3) -> List[Dict]:
    for attempt in range(retries):
        try:
            pdf_text = pdf_text.strip()
            if len(pdf_text.split()) < 3:  # تحذير لو النص قصير جدًا (أقل من 3 كلمات)
                print(f"Warning: Page {page_number} has very short text ('{pdf_text}'). Attempting to generate questions anyway...")

            print(f"Extracted PDF Text (Page {page_number}): {pdf_text}")

            # استخدام Gemini 2.0 Flash
            model = genai.GenerativeModel('gemini-2.0-flash')

            # تحسين الـ Prompt لاستغلال قدرات 2.0 Flash
            prompt = (
                f"Generate educational questions to assess understanding of the module based on the following text. "
                f"Include only multiple choice (with exactly 4 options) and true/false questions. "
                f"Ensure questions cover a diverse range of key topics such as Cloud Computing, Networking, Data Center Networks, and Virtualization. "
                f"If Data Center Networks is underrepresented, prioritize generating questions related to it when the text provides relevant context (e.g., LAN, SAN, or network infrastructure). "
                f"Apply advanced reasoning to link concepts across the text and avoid repetition or over-focusing on one topic; distribute questions evenly across the mentioned topics. "
                f"Do not include introductory text, explanations, or extra commentary before or after the questions. "
                f"For each question, provide the correct answer and explanation combined in one field, separated by a pipe symbol (e.g., 'Cloud Computing | The text states \"IT436 -Cloud Computing\"'), "
                f"and include difficulty level (easy, medium, hard) based on Bloom's Taxonomy:\n"
                f"- easy: Questions at Bloom's Remembering or Understanding levels, requiring recall of direct information or basic comprehension.\n"
                f"- medium: Questions at Bloom's Application or Analysis levels, requiring application of knowledge or linking concepts.\n"
                f"- hard: Questions at Bloom's Evaluation or Creating levels, requiring deep evaluation or creation of new ideas; use chain-of-thought reasoning for hard questions.\n"
                f"Strictly distribute the difficulty levels as follows: 40% easy, 40% medium, 20% hard, based on Bloom's Taxonomy levels. "
                f"If the text is very short or limited, generate at least two questions (one easy, one medium) based on any available information, even with inference, using the broader context of virtualization or related topics. "
                f"Format the response as a list where each question strictly follows this structure:\n"
                f"- For Multiple Choice: Question text? (Multiple Choice: option1, option2, option3, option4 | CorrectAnswer: answer|explanation | Difficulty: difficulty_level)\n"
                f"- For True/False: Question text? (True/False | CorrectAnswer: True or False|explanation | Difficulty: difficulty_level)\n"
                f"Ensure strict adherence to this format with proper separators (|, parentheses, etc.). Ensure all options in multiple-choice questions are clean, complete, and error-free. "
                f"Append a question? at the end of every question text. "
                f"{f'Focus on generating questions with difficulty level {difficulty_level} if specified.' if difficulty_level else ''}\n\n"
                f"Text:\n{pdf_text}"
            )

            # توليد الأسئلة باستخدام 2.0 Flash
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 300,  # زيادة لتوليد أسئلة أكتر
                    "temperature": 0.7
                }
            )
            print(f"Gemini Raw Response (Page {page_number}): {response.text}")

            raw_questions = response.text.strip()
            if not raw_questions:
                print(f"Gemini API returned an empty response for Page {page_number}. Skipping...")
                return []  # رجّع قايمة فاضية لو الرد فاضي

            questions = []
            for line in raw_questions.split("\n"):
                line = line.strip()
                if not line or not line.startswith("- "):
                    print(f"Skipping line in Page {page_number} due to incorrect format (not a question): {line}")
                    continue
                try:
                    parts = re.split(r'\s*\|\s*(?=CorrectAnswer:|Difficulty:)', line)
                    if len(parts) != 3:
                        print(f"Skipping line in Page {page_number} due to incorrect format (expected 3 parts, got {len(parts)}): {line}")
                        continue

                    if "Multiple Choice" in line:
                        question_part = parts[0].strip()
                        correct_answer_part = parts[1].replace("CorrectAnswer: ", "").strip()
                        difficulty = parts[2].replace("Difficulty: ", "").replace(")", "").strip()

                        # إذا كان Difficulty فارغ، نضيف "easy" افتراضيًا
                        if not difficulty:
                            difficulty = "easy"
                            print(f"Difficulty missing in Page {page_number} for question: {line}. Setting to 'easy' by default.")

                        answer, explanation = correct_answer_part.split("|", 1) if "|" in correct_answer_part else (correct_answer_part, "")
                        answer = clean_text(answer)
                        explanation = clean_text(explanation)

                        question_text = question_part.split("(Multiple Choice")[0].strip().replace("- ", "")
                        question_text = clean_text(question_text) + "?"  # إضافة علامة استفهام
                        options_part = question_part.split("(Multiple Choice: ")[1].replace(")", "").split(", ")
                        if len(options_part) != 4:
                            print(f"Skipping Multiple Choice question in Page {page_number} due to incorrect number of options (expected 4, got {len(options_part)}): {line}")
                            continue

                        cleaned_options = [clean_text(option.strip()) for option in options_part]

                        # تحقق إن الإجابة الصحيحة موجودة في الخيارات
                        if answer not in cleaned_options:
                            print(f"Correct answer '{answer}' not found in options for question in Page {page_number}. Adjusting options...")
                            if len(cleaned_options) >= 4:
                                cleaned_options.pop()  # حذف آخر خيار
                            cleaned_options.append(answer)  # إضافة الإجابة الصحيحة

                        questions.append({
                            "text": question_text,
                            "type": "MultipleChoice",
                            "correctAnswer": answer,
                            "explanation": explanation,
                            "difficulty": difficulty,
                            "options": [{"text": option} for option in cleaned_options],
                            "page": str(page_number),
                            "quiz_id": "quiz_001"
                        })
                    elif "True/False" in line:
                        question_part = parts[0].strip()
                        correct_answer_part = parts[1].replace("CorrectAnswer: ", "").strip()
                        difficulty = parts[2].replace("Difficulty: ", "").replace(")", "").strip()

                        # إذا كان Difficulty فارغ، نضيف "easy" افتراضيًا
                        if not difficulty:
                            difficulty = "easy"
                            print(f"Difficulty missing in Page {page_number} for question: {line}. Setting to 'easy' by default.")

                        answer, explanation = correct_answer_part.split("|", 1) if "|" in correct_answer_part else (correct_answer_part, "")
                        answer = clean_text(answer)
                        explanation = clean_text(explanation)

                        question_text = question_part.split("(True/False")[0].strip().replace("- ", "")
                        question_text = clean_text(question_text) + "?"  # إضافة علامة استفهام

                        questions.append({
                            "text": question_text,
                            "type": "TrueFalse",
                            "correctAnswer": answer,
                            "explanation": explanation,
                            "difficulty": difficulty,
                            "options": [{"text": "True"}, {"text": "False"}],
                            "page": str(page_number),
                            "quiz_id": "quiz_001"
                        })
                except Exception as e:
                    print(f"Error parsing line in Page {page_number}: {line}, Error: {str(e)}")
                    continue

            return questions  # رجّع الأسئلة حتى لو فاضية

        except Exception as e:
            if "429" in str(e) and "exceeded your current quota" in str(e).lower():
                retry_delay = 23
                print(f"Quota exceeded for Page {page_number}. Retrying after {retry_delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(retry_delay)
                continue
            elif "API key" in str(e).lower():
                raise HTTPException(status_code=500, detail=f"Gemini API error: Invalid or unauthorized API key. Please check your GEMINI_API_KEY: {str(e)}")
            else:
                print(f"Unexpected error while generating questions for Page {page_number}: {str(e)}. Skipping...")
                return []  # رجّع قايمة فاضية في حالة أي خطأ

    print(f"Failed to generate questions for Page {page_number} after {retries} attempts due to quota limits. Skipping...")
    return []  # رجّع قايمة فاضية لو فشل بعد كل المحاولات

@app.post("/generate-questions")
async def generate_questions(file: UploadFile = File(None), page: int = Form(None), text: str = Form(None), difficulty_level: str = Form(None)):
    try:
        if not file and not text:
            raise HTTPException(status_code=400, detail="Either a PDF file or text input is required.")
        
        if file and not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are supported.")

        pdf_reader = None
        if file:
            pdf_reader = PyPDF2.PdfReader(file.file)
            if len(pdf_reader.pages) == 0:
                raise HTTPException(status_code=400, detail="The PDF file is empty or corrupted.")

        all_questions = []

        if text:
            print(f"Manually Provided Text: {text}")
            questions = generate_questions_from_pdf(text, 1, difficulty_level)
            all_questions.extend(questions)
        elif page is not None:
            if page < 1 or page > len(pdf_reader.pages):
                raise HTTPException(status_code=400, detail=f"Invalid page number. Must be between 1 and {len(pdf_reader.pages)}.")
            text = pdf_reader.pages[page - 1].extract_text()
            print(f"Extracted PDF Text Before Processing (Page {page}): {text}")
            questions = generate_questions_from_pdf(text, page, difficulty_level)
            all_questions.extend(questions)
        else:
            for page_num in range(len(pdf_reader.pages)):
                text = pdf_reader.pages[page_num].extract_text()
                print(f"Extracted PDF Text Before Processing (Page {page_num + 1}): {text}")
                questions = generate_questions_from_pdf(text, page_num + 1, difficulty_level)
                all_questions.extend(questions)

        # لو مافيش أسئلة خالص بعد كل الصفحات، ارمي خطأ
        if not all_questions:
            raise HTTPException(status_code=400, detail="No valid questions were generated from the provided PDF. The content may be insufficient or not suitable for question generation.")

        # Comment out the .NET endpoint call for now
        # dotnet_response = send_questions_to_dotnet(all_questions)
        # print(f".NET Response: {dotnet_response}")

        # Save the questions to a file as a backup
        with open("questions_backup.json", "w", encoding="utf-8") as f:
            json.dump(all_questions, f, indent=4)

        return {
            "questions": all_questions,
            "message": "Questions generated successfully. Waiting for .NET endpoint URL to send the data."
        }

    except PyPDF2.errors.PdfReadError as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: The PDF file is corrupted or not readable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while processing the PDF: {str(e)}")