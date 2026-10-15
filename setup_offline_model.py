"""تنزيل وتثبيت نموذج Argos للإنجليزية إلى العربية (مطلوب مرة واحدة فقط)."""

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
INSTALLED_MODEL_DIR = MODEL_DIR / "installed"
MODEL_PATH = INSTALLED_MODEL_DIR / "en_ar"
MODEL_URLS = (
    "https://argos-net.com/v1/translate-en_ar-1_0.argosmodel",
    "https://data.argosopentech.com/argospm/v1/translate-en_ar-1_0.argosmodel",
)


def model_is_installed():
    try:
        metadata = json.loads((MODEL_PATH / "metadata.json").read_text(encoding="utf-8"))
        return (
            metadata.get("from_code") == "en"
            and metadata.get("to_code") == "ar"
            and (MODEL_PATH / "model" / "model.bin").is_file()
            and (MODEL_PATH / "sentencepiece.model").is_file()
        )
    except Exception:
        return False


def download_model(destination):
    last_error = None
    for url in MODEL_URLS:
        try:
            print(f"تنزيل نموذج الترجمة من المصدر الرسمي: {url}")
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                expected_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded += len(chunk)
                        if expected_size:
                            percent = downloaded * 100 // expected_size
                            print(f"\rالتنزيل: {percent}%", end="", flush=True)
                print()
            if destination.stat().st_size < 10 * 1024 * 1024:
                raise RuntimeError("حجم الملف الذي نُزّل أصغر من المتوقع.")
            return
        except Exception as error:
            last_error = error
            destination.unlink(missing_ok=True)
            print(f"تعذر التنزيل من هذا الرابط: {error}")
    raise RuntimeError(f"فشل تنزيل النموذج. تحقق من الإنترنت ثم أعد المحاولة. ({last_error})")


def extract_model(archive_path):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    INSTALLED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        raise RuntimeError(
            f"يوجد مجلد نموذج غير مكتمل: {MODEL_PATH}. أعد تسميته أو انقله ثم أعد الإعداد."
        )

    staging_parent = Path(tempfile.mkdtemp(prefix=".argos-stage-", dir=MODEL_DIR))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                resolved_target = (staging_parent / member.filename).resolve()
                if not resolved_target.is_relative_to(staging_parent.resolve()):
                    raise RuntimeError("يحتوي ملف النموذج على مسار غير صالح.")
                archive.extract(member, staging_parent)

        package_directory = None
        for metadata_file in staging_parent.rglob("metadata.json"):
            candidate = metadata_file.parent
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                metadata.get("from_code") == "en"
                and metadata.get("to_code") == "ar"
                and (candidate / "model" / "model.bin").is_file()
                and (candidate / "sentencepiece.model").is_file()
            ):
                package_directory = candidate
                break

        if package_directory is None:
            raise RuntimeError("الملف لا يحتوي نموذجًا صالحًا للترجمة من الإنجليزية إلى العربية.")

        shutil.move(str(package_directory), str(MODEL_PATH))
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def main():
    if model_is_installed():
        print("نموذج الإنجليزية إلى العربية مثبت محليًا بالفعل؛ لا يلزم تنزيله مرة أخرى.")
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="argos-download-") as temporary_directory:
        model_file = Path(temporary_directory) / "translate-en_ar-1_0.argosmodel"
        try:
            download_model(model_file)
            extract_model(model_file)
        except Exception as error:
            print(f"فشل إعداد النموذج: {error}")
            return 1

    if not model_is_installed():
        print("لم يكتمل تثبيت نموذج en-to-ar.")
        return 1

    print("اكتمل تثبيت نموذج en-to-ar محليًا. يمكنك الآن فصل الإنترنت وتشغيل التطبيق.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
