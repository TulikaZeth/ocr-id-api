from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import cv2
import numpy as np
import re
from collections import Counter
import base64
import io
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app) 

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize OCR reader (do this once for better performance)
ocr_reader = easyocr.Reader(['en'], gpu=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """Enhance image quality for better OCR results"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.2, beta=10)
    return enhanced

def extract_library_id(text):
    """Extract library ID using multiple patterns"""
    patterns = [
        r'\b\d{4,}[A-Za-z]{2,}\d{2,}\b',
        r'\b\d{2,}[A-Za-z]{1,}\d{4,}\b',
        r'\b[A-Za-z]{2,}\d{6,}\b',
        r'\b\d{6,}[A-Za-z]{1,}\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

def extract_name_multiple_strategies(text):
    """Try multiple strategies to extract name"""
    ignore_keywords = [
        "kiet", "ghaziabad", "connecting", "group", "institutions",
        "www.kiet.edu", "delhi", "ncr", "learning", "card", "valid", "upto",
        "college", "university", "student", "identity"
    ]
    
    # Strategy 1: Relationship indicators
    relationship_patterns = [
        r'(.+?)\s+(?:S/O|D/O|SIO|DIO|S\.O\.|D\.O\.|SON OF|DAUGHTER OF)',
        r'(.+?)\s+(?:W/O|WIO|WIFE OF)',
    ]
    
    for pattern in relationship_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name_candidate = match.group(1).strip()
            name_candidate = re.sub(r'[^\w\s]', ' ', name_candidate)
            name_candidate = ' '.join(name_candidate.split())
            
            words = name_candidate.split()
            if 1 <= len(words) <= 4 and 3 <= len(name_candidate) <= 50:
                filtered_words = [w for w in words if w.lower() not in ignore_keywords]
                if filtered_words:
                    return ' '.join(filtered_words).title()
    
    # Strategy 2: Capitalized sequences
    cap_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',
        r'\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,3}\b',
        r'\b[A-Z][A-Z\s]{3,20}[A-Z]\b',
    ]
    
    for pattern in cap_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if not any(keyword in match.lower() for keyword in ignore_keywords):
                words = match.split()
                if 1 <= len(words) <= 4:
                    return match.title()
    
    return None

def process_image(image_data):
    """Process image and extract data"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Run OCR
        ocr_results = ocr_reader.readtext(processed_image)
        
        # Extract raw text
        raw_text = "\n".join([text for _, text, confidence in ocr_results if confidence > 0.3])
        
        # Initialize data structure
        extracted_data = {
            "name": "N/A",
            "collegeLibraryId": "N/A", 
            "branchName": "N/A",
            "graduationYear": "N/A",
            "course": "N/A"
        }
        
        # Extract Library ID
        library_id = extract_library_id(raw_text)
        if library_id:
            extracted_data["collegeLibraryId"] = library_id
            
            # Extract branch
            branch_patterns = [
                r'\d+([A-Za-z]{2,})\d+',
                r'([A-Za-z]{2,})\d+$',
                r'^([A-Za-z]{2,})',
            ]
            
            for pattern in branch_patterns:
                branch_match = re.search(pattern, library_id)
                if branch_match:
                    extracted_data["branchName"] = branch_match.group(1).upper()
                    break
        
        # Extract Name
        extracted_name = extract_name_multiple_strategies(raw_text)
        if extracted_name:
            extracted_data["name"] = extracted_name
        
        # Extract Graduation Year
        year_matches = re.findall(r'\b(20\d{2})\b', raw_text)
        current_year = 2025
        plausible_years = [int(y) for y in year_matches if 2015 <= int(y) <= current_year + 6]
        
        if plausible_years:
            year_counts = Counter(plausible_years)
            extracted_data["graduationYear"] = str(year_counts.most_common(1)[0][0])
        
        # Extract Course
        course_patterns = [
            r'\b(B\.?TECH|BTECH|B\.?E|BE|M\.?TECH|MTECH|MBA|MCA|BCA)\b',
            r'\b(BACHELOR|MASTER|DIPLOMA)\b',
        ]
        
        for pattern in course_patterns:
            course_match = re.search(pattern, raw_text, re.IGNORECASE)
            if course_match:
                extracted_data["course"] = course_match.group(0).upper()
                break
        
        return {
            "success": True,
            "data": extracted_data,
            "raw_text": raw_text,
            "confidence_scores": [{"text": text, "confidence": float(conf)} for _, text, conf in ocr_results]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "OCR ID Extractor"})

@app.route('/api/extract-id', methods=['POST'])
def extract_id_data():
    """Extract ID data from uploaded image file"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type"}), 400
        
        # Read and process image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400
        
        # Process image and extract data
        result = process_image(image)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route('/api/extract-id-base64', methods=['POST'])
def extract_id_from_base64():
    """Extract ID data from base64 encoded image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No base64 image data provided"}), 400
        
        # Decode base64 image
        base64_str = data['image']
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        image_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400
        
        # Process image and extract data
        result = process_image(image)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "File too large"}), 413

if __name__ == '__main__':
    print("🚀 Starting OCR ID Extractor API...")
    print("📋 Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/extract-id - Upload image file")
    print("  POST /api/extract-id-base64 - Send base64 image")
    app.run(debug=True, host='0.0.0.0', port=5000)