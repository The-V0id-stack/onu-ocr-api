import os
import re
import hashlib
import pytesseract
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Habilitar CORS
from PIL import Image

# Ruta de Tesseract en tu sistema (ajusta la ruta si es necesario)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configuración de Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

app.secret_key = "secret_key"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función para verificar si el archivo es válido
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para preprocesar la imagen para mejor OCR
def preprocess_image(image_path: str) -> Image:
    # Leer la imagen
    img = cv2.imread(image_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mejorar el contraste de la imagen
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)  # Aumentar el contraste

    # Aplicar un filtro bilateral para reducir el ruido y preservar los bordes
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Aplicar umbralización adaptativa para mejorar la imagen en condiciones de iluminación no uniforme
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Aumentar el tamaño de la imagen para mejorar la precisión del OCR
    large_img = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convertir la imagen a formato PIL para ser procesada por pytesseract
    pil_img = Image.fromarray(large_img)
    return pil_img

# Función para extraer el PON S/N
def extract_sn_from_text(text: str) -> str | None:
    """
    Extrae el PON S/N de la imagen OCR.
    """
    text_upper = text.upper()
    lines = [l.strip() for l in text_upper.splitlines() if l.strip()]
    
    # Ajustar la expresión regular para ser más flexible y captar diferentes variaciones
    for line in reversed(lines):
        if "GPON" in line:
            sn = re.sub(r"[^A-Z0-9]", "", line)  # Filtra caracteres no alfanuméricos
            if len(sn) >= 8:  # Asegurarnos de que el S/N tenga una longitud mínima
                return sn
    return None

# Función para leer el QR de la imagen usando OpenCV
def read_qr_code(image_path: str):
    img = cv2.imread(image_path)
    detector = cv2.QRCodeDetector()
    value, pts, qr_code = detector(img)
    if value:
        return value
    return None

# Función para adivinar la marca basada en el PON S/N
def guess_brand(sn: str) -> str:
    """
    Adivina la marca basada en el PON S/N.
    """
    if sn:
        sn = sn.upper()

        # Reglas simples basadas en el S/N
        if "R" in sn or sn.isdigit():
            return "OpticTimes"
        elif "VSOL" in sn or sn.startswith("V"):
            return "VSOL"
        else:
            return "Marca Desconocida"
    return "Marca Desconocida"

# Endpoint para recibir imágenes y procesarlas
@app.route("/api/ocr", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected image"}), 400

    if file and allowed_file(file.filename):
        # Guardar archivo con un nombre único
        filename = secure_filename(file.filename)
        file_hash = hashlib.sha256(file.read()).hexdigest()
        file.seek(0)  # Rewind after reading for the hash
        saved_filename = f"{file_hash}.jpg"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
        file.save(save_path)

        # Intentar leer el QR primero usando OpenCV
        qr_data = read_qr_code(save_path)

        if qr_data:
            # Si se lee el QR, devolver su valor
            return jsonify({
                "sn": qr_data,
                "brand": "QR Detected",
                "image_url": f"/uploads/{saved_filename}"
            })

        # Preprocesamos la imagen para obtener el texto OCR
        pil_img = preprocess_image(save_path)
        text = pytesseract.image_to_string(pil_img, config='--psm 6')

        # Extraemos el PON S/N y la marca
        sn = extract_sn_from_text(text)
        brand = guess_brand(sn)

        return jsonify({
            "sn": sn,
            "brand": brand,
            "image_url": f"/uploads/{saved_filename}"
        })

    return jsonify({"error": "Invalid file format"}), 400

# Endpoint para servir las imágenes subidas
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Inicia la aplicación
if __name__ == "__main__":
    app.run(debug=True)
