import os
import json
import queue
import re
import shutil
import threading
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import tkinter as tk
from mss import MSS


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "installed" / "en_ar"

try:
    import ctranslate2
    import sentencepiece as spm
except ImportError:
    ctranslate2 = None
    spm = None


# يمكن تخصيصه بمتغير البيئة TESSERACT_CMD، أو يكتشف البرنامج المسار تلقائيًا.
def find_tesseract():
    configured_path = os.environ.get("TESSERACT_CMD")
    candidates = [
        configured_path,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


TESSERACT_PATH = find_tesseract()
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# نسبة إلى الشاشة الأساسية: تلتقط شريط النص قرب أسفل الشاشة.
REGION = {"x": 0.05, "y": 0.78, "width": 0.90, "height": 0.18}
CAPTURE_INTERVAL = 1.2
SIMILARITY_THRESHOLD = 0.94


def get_capture_region(monitor):
    """حوّل النسب إلى إحداثيات صالحة داخل مساحة الشاشة الأساسية."""
    monitor_width = monitor["width"]
    monitor_height = monitor["height"]
    x = min(max(round(REGION["x"] * monitor_width), 0), monitor_width - 1)
    y = min(max(round(REGION["y"] * monitor_height), 0), monitor_height - 1)
    width = min(max(round(REGION["width"] * monitor_width), 1), monitor_width - x)
    height = min(max(round(REGION["height"] * monitor_height), 1), monitor_height - y)
    return {
        "left": monitor["left"] + x,
        "top": monitor["top"] + y,
        "width": width,
        "height": height,
    }


def process_video_frame(screenshot):
    image = np.asarray(screenshot)
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def similar(first, second):
    return SequenceMatcher(None, first, second).ratio()


def offline_model_ready():
    """تحقق من ملفات النموذج محليًا دون أي اتصال خارجي."""
    if ctranslate2 is None or spm is None:
        return False
    try:
        metadata = json.loads(
            (MODEL_PATH / "metadata.json").read_text(encoding="utf-8")
        )
        return (
            metadata.get("from_code") == "en"
            and metadata.get("to_code") == "ar"
            and (MODEL_PATH / "model" / "model.bin").is_file()
            and (MODEL_PATH / "sentencepiece.model").is_file()
        )
    except Exception:
        return False


class LocalArgosTranslator:
    """يشغّل نموذج Argos مباشرةً عبر CTranslate2 وSentencePiece محليًا."""

    def __init__(self):
        self.model = ctranslate2.Translator(
            str(MODEL_PATH / "model"), device="cpu"
        )
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=str(MODEL_PATH / "sentencepiece.model")
        )

    def translate(self, text):
        # النصوص المرئية قصيرة عادةً؛ نقسم الجمل بعلامات ترقيم محلية بلا نماذج إضافية.
        pieces = [
            piece.strip()
            for piece in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
            if piece.strip()
        ]
        return " ".join(self._translate_piece(piece) for piece in pieces)

    def _translate_piece(self, text):
        tokens = self.tokenizer.encode(text, out_type=str)
        if not tokens:
            return ""
        result = self.model.translate_batch(
            [tokens],
            replace_unknowns=True,
            max_batch_size=32,
            batch_type="tokens",
            beam_size=4,
            num_hypotheses=1,
            length_penalty=0.2,
            return_scores=True,
        )[0]
        return self.tokenizer.decode(result.hypotheses[0]).replace("_", " ").lstrip()


class ScreenTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.worker_thread = None
        self.stop_event = None
        self.previous_text = ""
        self.worker_failed = False

        root.title("المترجم المباشر للشاشة — دون إنترنت")
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.90)
        root.configure(bg="black")
        root.geometry(self._window_geometry())
        root.protocol("WM_DELETE_WINDOW", self.close)

        top_frame = tk.Frame(root, bg="black")
        top_frame.pack(fill="x", pady=5)

        self.button = tk.Button(
            top_frame,
            text="▶ تشغيل الترجمة المحلية",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            cursor="hand2",
            command=self.toggle_translation,
        )
        self.button.pack(pady=2)

        self.label = tk.Label(
            root,
            text="ترجمة محلية دون إنترنت. اضغط تشغيل بعد إظهار النص على الشاشة.",
            font=("Arial", 14, "bold"),
            fg="#00FF00",
            bg="black",
            wraplength=780,
            justify="center",
        )
        self.label.pack(expand=True, fill="both", padx=10, pady=4)
        self.root.after(100, self.poll_events)

    def _window_geometry(self):
        width, height = 800, 140
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = max(0, (screen_width - width) // 2)
            y = max(10, screen_height - height - 240)
            return f"{width}x{height}+{x}+{y}"
        except tk.TclError:
            return f"{width}x{height}+100+100"

    def toggle_translation(self):
        if self.worker_thread and self.worker_thread.is_alive():
            if self.stop_event and not self.stop_event.is_set():
                self.stop_event.set()
                self.button.config(state="disabled")
                self.label.config(text="جارٍ إيقاف الالتقاط...", fg="#FFD54F")
            return

        if not TESSERACT_PATH:
            self.label.config(
                text=(
                    "لم يتم العثور على Tesseract OCR. ثبّته ثم أعد التشغيل، "
                    "أو عيّن TESSERACT_CMD إلى مساره."
                ),
                fg="#FF8A80",
            )
            return

        if ctranslate2 is None or spm is None:
            self.label.config(
                text="مكتبة الترجمة المحلية غير مثبتة. شغّل: python -m pip install -r requirements.txt",
                fg="#FF8A80",
            )
            return

        if not offline_model_ready():
            self.label.config(
                text="نموذج الترجمة غير مثبت. شغّل مرة واحدة: python setup_offline_model.py",
                fg="#FFB74D",
            )
            return

        self.previous_text = ""
        self.worker_failed = False
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self.capture_and_translate,
            args=(self.stop_event,),
            daemon=True,
        )
        self.button.config(
            text="■ إيقاف الترجمة", bg="#E53935", state="normal"
        )
        self.label.config(text="بدأ الالتقاط والترجمة المحلية.", fg="#00FF00")
        self.worker_thread.start()

    def capture_and_translate(self, stop_event):
        try:
            translator = LocalArgosTranslator()

            with MSS() as screen:
                if len(screen.monitors) < 2:
                    raise RuntimeError("لم يتم العثور على شاشة يمكن التقاطها.")
                region = get_capture_region(screen.monitors[1])

                while not stop_event.is_set():
                    try:
                        screenshot = screen.grab(region)
                        processed_image = process_video_frame(screenshot)
                        text = pytesseract.image_to_string(
                            processed_image,
                            lang="eng",
                            config="--oem 3 --psm 6",
                        )
                        text = " ".join(text.split())

                        if len(text) > 3 and re.search(r"[a-zA-Z]", text):
                            if not self.previous_text or similar(text, self.previous_text) < SIMILARITY_THRESHOLD:
                                # استدعاء نموذج مثبت محليًا؛ لا يستخدم الإنترنت.
                                translated_text = translator.translate(text)
                                if translated_text:
                                    self.previous_text = text
                                    self.events.put(("translation", translated_text))
                    except Exception as error:
                        self.events.put(("error", f"خطأ في الالتقاط أو الترجمة المحلية: {error}"))
                        if stop_event.wait(2):
                            break
                        continue

                    if stop_event.wait(CAPTURE_INTERVAL):
                        break
        except Exception as error:
            self.worker_failed = True
            self.events.put(("fatal", f"تعذر بدء الالتقاط أو تحميل نموذج الترجمة: {error}"))
        finally:
            self.events.put(("finished", None))

    def poll_events(self):
        try:
            while True:
                event, message = self.events.get_nowait()
                if event == "translation":
                    self.label.config(text=message, fg="#00FF00")
                elif event == "error":
                    self.label.config(text=message, fg="#FFB74D")
                elif event == "fatal":
                    self.worker_failed = True
                    self.label.config(text=message, fg="#FF8A80")
                elif event == "finished":
                    self.worker_thread = None
                    self.stop_event = None
                    self.button.config(
                        text="▶ تشغيل الترجمة المحلية", bg="#4CAF50", state="normal"
                    )
                    if not self.worker_failed:
                        self.label.config(text="الترجمة متوقفة.", fg="#00FF00")
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(100, self.poll_events)

    def close(self):
        if self.stop_event:
            self.stop_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    ScreenTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
