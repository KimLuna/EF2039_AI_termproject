import os
import warnings
import logging
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from spleeter.separator import Separator

import librosa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display

# ===== 기존 main.py 설정 그대로 가져오기 =====
if "TF_CONFIG" in os.environ:
    del os.environ["TF_CONFIG"]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("spleeter").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "waveforms").mkdir(exist_ok=True)
(STATIC_DIR / "spectrograms").mkdir(exist_ok=True)

# ===== Spleeter 모델 캐싱 =====
_separators = {}

def get_separator(model_type: str) -> Separator:
    if model_type not in _separators:
        model_name = f"spleeter:{model_type}"
        _separators[model_type] = Separator(model_name)
    return _separators[model_type]

# ===== 분석 & 시각화 함수 =====

def generate_waveform_image(audio_path: Path, image_path: Path):
    y, sr = librosa.load(str(audio_path), mono=True)
    plt.figure(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

def generate_spectrogram_image(audio_path: Path, image_path: Path):
    y, sr = librosa.load(str(audio_path), mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(8, 3))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

def analyze_stem(stem_path: Path) -> dict:
    y, sr = librosa.load(str(stem_path), mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = float(librosa.feature.rms(y=y).mean())
    peak = float(np.max(np.abs(y)))

    wave_img = STATIC_DIR / "waveforms" / f"{stem_path.stem}_wave.png"
    spec_img = STATIC_DIR / "spectrograms" / f"{stem_path.stem}_spec.png"

    generate_waveform_image(stem_path, wave_img)
    generate_spectrogram_image(stem_path, spec_img)

    return {
        "name": stem_path.name,
        "duration": round(duration, 2),
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "relpath": str(stem_path.relative_to(OUTPUT_DIR)),
        "wave_img": f"waveforms/{wave_img.name}",
        "spec_img": f"spectrograms/{spec_img.name}",
    }

def separate_audio_web(input_path: Path, model_type: str) -> Path:
    separator = get_separator(model_type)
    file_stem = input_path.stem
    out_dir = OUTPUT_DIR / file_stem / model_type

    separator.separate_to_file(
        str(input_path),
        str(out_dir),
        filename_format="{instrument}.wav"
    )

    return out_dir

# ===== Flask 앱 =====
app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    template_folder=str(BASE_DIR / "templates")
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", stems=None)

    file = request.files.get("file")
    model_type = request.form.get("model", "2stems")

    if not file or file.filename == "":
        return render_template("index.html", error="파일을 선택해주세요.", stems=None)

    upload_path = UPLOAD_DIR / file.filename
    file.save(upload_path)

    try:
        out_dir = separate_audio_web(upload_path, model_type)
    except Exception as e:
        return render_template("index.html", error=f"Spleeter 오류: {e}", stems=None)

    stems = []
    for stem_file in sorted(out_dir.glob("*.wav")):
        stems.append(analyze_stem(stem_file))

    return render_template(
        "index.html",
        stems=stems,
        original_filename=file.filename,
        model_type=model_type,
        error=None,
    )

@app.route("/download/<path:relpath>")
def download(relpath):
    full_path = OUTPUT_DIR / relpath
    if not full_path.exists():
        return "File not found", 404

    return send_from_directory(
        directory=str(full_path.parent),
        path=full_path.name,
        as_attachment=False,      
        mimetype="audio/wav"      
    )

    @app.route("/audio/<path:relpath>")
    def serve_audio(relpath):
        full_path = OUTPUT_DIR / relpath
        return send_from_directory(
            directory=str(full_path.parent),
            path=full_path.name,
            mimetype="audio/wav"
        )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
