from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import requests
import PyPDF2
import google.generativeai as genai
from typing import List, Dict
import os
from dotenv import load_dotenv

# تحديد المسار بتاع ملف .env بشكل صريح
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
print("Loaded .env file successfully")  # Debugging
print(f"GEMINI_API_KEY from env: {os.getenv('GEMINI_API_KEY')}")  # Debugging

app = FastAPI()

# إعداد Gemini API باستخدام Environment Variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)

# إعداد رابط .NET لو متاح في .env
DOT_NET_API_URL = os.getenv("DOT_NET_API_URL", "https://their-api.com/api/questions/store-questions")

# دالة لتوليد أسئلة باستخدام Gemini API
def generate_questions_from_pdf(pdf_text: str, page_number: int) -> List[Dict]:
    try:
        if not pdf_text.strip():
            raise ValueError("Extracted PDF text is empty")

        model = genai.GenerativeModel('gemini-2.0-flash')  # افترضنا إنك بتستخدمي Gemini 2.0 Flash
        prompt = (
            f"Generate educational questions from the following text. "
            f"Include only multiple choice (with 4 options) and true/false questions. "
            f"For each question, provide the correct answer, an explanation, and difficulty level (easy, medium, hard). "
            f"Format the response as a list where each question has the following structure:\n"
            f"- For Multiple Choice: Question text (Multiple Choice: option1, option2, option3, option4 | Correct: correct_option | Explanation: explanation_text | Difficulty: difficulty_level)\n"
            f"- For True/False: Question text (True/False | Correct: True or False | Explanation: explanation_text | Difficulty: difficulty_level)\n\n"
            f"Text:\n{pdf_text}"
        )
        response = model.generate_content(prompt)

        # افترضنا إن Gemini API بيرجّع الأسئلة كنص، هنحوّله لـ JSON
        raw_questions = response.text.strip()
        if not raw_questions:
            raise ValueError("Gemini API returned empty response")

        # تحويل النص لـ JSON متوافق مع .NET Model
        questions = []
        for line in raw_questions.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "Multiple Choice" in line:
                try:
                    question_text = line.split("(Multiple Choice")[0].strip().replace("1. ", "").replace("2. ", "")
                    options_part = line.split("(Multiple Choice: ")[1].split(" | Correct: ")[0].replace(")", "").split(", ")
                    correct_answer = line.split(" | Correct: ")[1].split(" | Explanation: ")[0].strip()
                    explanation = line.split(" | Explanation: ")[1].split(" | Difficulty: ")[0].strip()
                    difficulty = line.split(" | Difficulty: ")[1].replace(")", "").strip()
                    if len(options_part) != 4:
                        continue  # بنتأكد إن الخيارات 4
                    # دمج الـ explanation و difficulty مع correctAnswer في string واحد
                    combined_correct_answer = f"{correct_answer} (Explanation: {explanation}, Difficulty: {difficulty})"
                    questions.append({
                        "text": question_text,
                        "type": "MultipleChoice",
                        "correctAnswer": combined_correct_answer,  # string واحد
                        "options": [{"text": option.strip()} for option in options_part],
                        "quizId": 1,  # قيمة مؤقتة
                        "page": str(page_number)  # حقل page كـ string بناءً على رقم الصفحة
                    })
                except:
                    continue
            elif "True/False" in line:
                try:
                    question_text = line.split("(True/False")[0].strip().replace("1. ", "").replace("2. ", "")
                    correct_answer = line.split(" | Correct: ")[1].split(" | Explanation: ")[0].strip()
                    explanation = line.split(" | Explanation: ")[1].split(" | Difficulty: ")[0].strip()
                    difficulty = line.split(" | Difficulty: ")[1].replace(")", "").strip()
                    # دمج الـ explanation و difficulty مع correctAnswer في string واحد
                    combined_correct_answer = f"{correct_answer} (Explanation: {explanation}, Difficulty: {difficulty})"
                    # بالنسبة لـ True/False، بنحط الخيارات كـ True و False بس
                    questions.append({
                        "text": question_text,
                        "type": "TrueFalse",
                        "correctAnswer": combined_correct_answer,  # string واحد
                        "options": [{"text": "True"}, {"text": "False"}],
                        "quizId": 1,  # قيمة مؤقتة
                        "page": str(page_number)  # حقل page كـ string بناءً على رقم الصفحة
                    })
                except:
                    continue

        if not questions:
            raise ValueError("No valid questions generated from Gemini API")
        return questions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions with Gemini API: {str(e)}")

@app.post("/generate-questions")
async def generate_questions(file: UploadFile = File(...), page: int = None, userId: str = Form(...)):
    try:
        # اقرأ الـ PDF
        pdf_reader = PyPDF2.PdfReader(file.file)
        all_questions = []

        if page is not None:
            # لو المستخدم حدّد صفحة معيّنة
            if page < 1 or page > len(pdf_reader.pages):
                raise HTTPException(status_code=400, detail="Invalid page number")
            text = pdf_reader.pages[page - 1].extract_text()
            questions = generate_questions_from_pdf(text, page)  # بنمرر رقم الصفحة
            all_questions.extend(questions)
        else:
            # لو المستخدم ما حدّدش صفحة، نولّد أسئلة من كل صفحة
            for page_num in range(len(pdf_reader.pages)):
                text = pdf_reader.pages[page_num].extract_text()
                questions = generate_questions_from_pdf(text, page_num + 1)  # page_num + 1 عشان الصفحات تبدأ من 1
                all_questions.extend(questions)

        # إضافة userId لكل سؤال
        for question in all_questions:
            question["userId"] = userId

        # إرسال الأسئلة لـ .NET endpoint
        response = requests.post(DOT_NET_API_URL, json=all_questions)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to store questions in .NET API")

        # إرجاع الأسئلة لـ Angular
        return all_questions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")