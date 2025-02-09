from flask import Flask, render_template, request, jsonify
import os
from PyPDF2 import PdfReader  # Note the capital PyPDF2
from groq import Groq

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'abc'

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Groq client
groq_client = Groq(api_key="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")
# Test API connection at startup
try:
    test_response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": "Test connection"}],
        model="mixtral-8x7b-32768",
        temperature=0.1,
        max_tokens=10
    )
    print("✅ Groq API connection successful!")
except Exception as e:
    print(f"❌ Groq API connection failed: {str(e)}")
# Global variable to store PDF content
pdf_content = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global pdf_content
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Extract text from PDF
        try:
            reader = PdfReader(file_path)
            pdf_content = ""
            for page in reader.pages:
                pdf_content += page.extract_text()
            return jsonify({'message': 'PDF uploaded and processed successfully'})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/ask', methods=['POST'])
def ask_question():
    global pdf_content
    data = request.json
    question = data.get('question')

    if not pdf_content:
        return jsonify({'error': 'Please upload a PDF first'})

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Use the following PDF content to answer questions: {pdf_content}"
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)