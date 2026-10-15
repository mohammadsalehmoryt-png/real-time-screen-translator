import tkinter as tk
import threading
import time
from mss import MSS
import pytesseract
import cv2
import numpy as np
from deep_translator import GoogleTranslator
import re
from difflib import SequenceMatcher # إضافة مكتبة المقارنة الذكية

# مسار برنامج Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# إحداثيات مربع الترجمة (يجب عليك تصغيرها لاحقاً لتقليل التشويش)
REGION = {"top": 1000, "left": 450, "width": 920, "height": 150}

PREV_TEXT = ""
is_translating = False

def process_video_frame(screenshot):
    img_np = np.array(screenshot)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def similar(a, b):
    """دالة تقارن بين النصين وتعطينا نسبة التشابه بينهما"""
    return SequenceMatcher(None, a, b).ratio()

def capture_and_translate(label):
    global PREV_TEXT, is_translating
    translator = GoogleTranslator(source='auto', target='ar')
    
    with MSS() as sct:
        while is_translating:
            try:
                screenshot = sct.grab(REGION)
                processed_img = process_video_frame(screenshot)
                
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(processed_img, lang='eng', config=custom_config).strip()
                text = " ".join(text.split())

                has_letters = bool(re.search('[a-zA-Z]', text))

                if text and len(text) > 3 and has_letters:
                    # الشرط السحري: إذا كان النص الجديد يشبه النص القديم بنسبة تزيد عن 60%
                    # فهذا يعني أنه نفس النص مع بعض التشويش، لذا لا نترجمه مرة أخرى!
                    if similar(text, PREV_TEXT) < 0.60:
                        print(f"النص المقروء الفعلي: {text}")
                        PREV_TEXT = text
                        
                        translated_text = translator.translate(text)
                        label.after(0, update_label, label, translated_text)
                
                time.sleep(1.2)
                
            except Exception as e:
                if "too many requests" in str(e).lower():
                    print("تنبيه: ننتظر 3 ثواني لتجنب حظر جوجل...")
                    time.sleep(3)
                else:
                    print(f"حدث خطأ: {e}")
                    time.sleep(1)

def update_label(label, text):
    label.config(text=text)

def toggle_translation(btn, label):
    global is_translating
    
    if not is_translating:
        is_translating = True
        btn.config(text="🛑 إيقاف الترجمة", bg="#FF4C4C", fg="white")
        label.config(text="جاري التقاط الشاشة والترجمة... (شغّل الفيديو الآن)")
        threading.Thread(target=capture_and_translate, args=(label,), daemon=True).start()
    else:
        is_translating = False
        btn.config(text="▶️ تشغيل الترجمة", bg="#4CAF50", fg="white")
        label.config(text="الترجمة متوقفة مؤقتاً.")

def setup_ui():
    root = tk.Tk()
    root.title("المترجم المباشر (Google Translate)")
    root.attributes("-topmost", True)
    root.geometry("800x130+550+750") 
    root.configure(bg='black')
    root.attributes("-alpha", 0.85)

    top_frame = tk.Frame(root, bg='black')
    top_frame.pack(fill='x', pady=5)

    label = tk.Label(
        root, 
        text="البرنامج جاهز.. رتب شاشتك ثم اضغط (تشغيل الترجمة) بالأعلى", 
        font=("Arial", 16, "bold"), 
        fg="#00FF00",
        bg="black",
        wraplength=780, 
        justify="center"
    )
    
    btn_toggle = tk.Button(
        top_frame, 
        text="▶️ تشغيل الترجمة", 
        font=("Arial", 12, "bold"), 
        bg="#4CAF50", 
        fg="white", 
        cursor="hand2",
        command=lambda: toggle_translation(btn_toggle, label)
    )
    btn_toggle.pack(pady=2)

    label.pack(expand=True, fill="both", padx=10, pady=2)

    root.mainloop()

if __name__ == "__main__":
    setup_ui()