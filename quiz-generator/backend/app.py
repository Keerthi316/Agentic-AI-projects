"""
Flask Backend Server for AI-Powered Quiz Generator
Handles PPT upload, AI quiz generation, and serves the frontend
"""

import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ppt_parser import extract_text_from_pptx, get_combined_text
from quiz_generator import generate_with_fallback

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
ALLOWED_EXTENSIONS = {'.pptx'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload_pptx():
    """
    Upload and parse a PPTX file.
    
    Returns:
        JSON with slide count and preview content
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only .pptx files are allowed'}), 400
        
        # Generate unique filename to prevent conflicts
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(file_path)
        
        # Parse the PPTX file
        result = extract_text_from_pptx(file_path)
        
        # Create preview (first 200 chars of each slide)
        slides_preview = []
        for slide in result['slides'][:3]:  # Preview first 3 slides
            preview_text = slide['text'][:200] + ('...' if len(slide['text']) > 200 else '')
            slides_preview.append({
                'slide_number': slide['slide_number'],
                'preview': preview_text,
                'has_content': slide['text'] != '(No text content)'
            })
        
        return jsonify({
            'file_id': unique_filename,
            'slide_count': result['slide_count'],
            'slides_preview': slides_preview,
            'message': f'Successfully uploaded {file.filename} with {result["slide_count"]} slides'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz():
    """
    Generate MCQs based on uploaded PPT content.
    
    Request body:
        file_id (str): The unique filename returned from upload
        num_questions (int): Number of questions (5-30)
        difficulty (str): Difficulty level (simple/medium/complex)
    
    Returns:
        JSON with generated quiz questions
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        file_id = data.get('file_id')
        num_questions = data.get('num_questions', 10)
        difficulty = data.get('difficulty', 'medium')
        
        if not file_id:
            return jsonify({'error': 'No file_id provided. Please upload a file first.'}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, file_id)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found. Please upload again.'}), 404
        
        # Validate num_questions
        try:
            num_questions = int(num_questions)
        except (ValueError, TypeError):
            return jsonify({'error': 'num_questions must be a number'}), 400
        
        if num_questions < 5 or num_questions > 30:
            return jsonify({'error': 'Number of questions must be between 5 and 30'}), 400
        
        # Validate difficulty
        valid_difficulties = ['simple', 'medium', 'complex']
        if difficulty not in valid_difficulties:
            return jsonify({'error': f'Difficulty must be one of: {", ".join(valid_difficulties)}'}), 400
        
        # Extract text from the uploaded PPTX
        slide_text = get_combined_text(file_path)
        
        # Validate that the PPT has meaningful content
        if not slide_text.strip() or slide_text.strip() == '':
            return jsonify({'error': 'No text content found in the presentation'}), 400
        
        # Generate quiz using AI
        questions = generate_with_fallback(slide_text, num_questions, difficulty)
        
        return jsonify({
            'questions': questions,
            'total_questions': len(questions),
            'difficulty': difficulty
        }), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to generate quiz: {str(e)}'}), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup_file():
    """
    Delete the uploaded file to free space.
    
    Request body:
        file_id (str): The unique filename to delete
    """
    try:
        data = request.get_json()
        file_id = data.get('file_id') if data else None
        
        if file_id:
            file_path = os.path.join(UPLOAD_FOLDER, file_id)
            if os.path.exists(file_path):
                os.remove(file_path)
                return jsonify({'message': 'File cleaned up'}), 200
        
        return jsonify({'message': 'Nothing to clean up'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)