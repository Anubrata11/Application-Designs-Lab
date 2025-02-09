import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize global variables
linear_model = LinearRegression()
lstm_model = None
scaler = StandardScaler()
mm_scaler = MinMaxScaler()
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_data_linear(df):
    """Prepare data for Linear Regression"""
    # Create features
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Create features and target
    X = df[['SMA_5', 'SMA_20', 'RSI', 'Price_Change', 'Volatility']].values
    y = df['Close'].values
    
    return X, y

def prepare_data_lstm(df, look_back=30):
    """Prepare data for LSTM"""
    # Ensure enough data points for look_back
    if len(df) < look_back:
        look_back = len(df) // 2

    # Scale the data
    scaled_data = mm_scaler.fit_transform(df[['Close']].values)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_lstm_model(input_shape):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_linear_model(X, y):
    """Train the linear regression model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    linear_model.fit(X_train_scaled, y_train)
    return linear_model.score(X_test_scaled, y_test)

def train_lstm_model(X, y):
    """Train the LSTM model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    global lstm_model
    lstm_model = create_lstm_model((X.shape[1], 1))
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    return lstm_model.evaluate(X_test, y_test)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_type = data.get('model_type', 'linear')
        
        # Load and prepare data
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        
        if model_type == 'linear':
            # Linear Regression prediction
            X, y = prepare_data_linear(df)
            accuracy = train_linear_model(X, y)
            last_data = scaler.transform([X[-1]])
            prediction = linear_model.predict(last_data)[0]
        else:
            # LSTM prediction
            X, y = prepare_data_lstm(df)
            accuracy = train_lstm_model(X, y)
            
            # Predict next point
            last_sequence = X[-1]  # Last sequence in the training data
            last_sequence = np.expand_dims(last_sequence, axis=0)
            prediction = lstm_model.predict(last_sequence)
            prediction = mm_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        
        response = {
            'prediction': float(prediction),
            'accuracy': float(1 - accuracy) if model_type == 'lstm' else float(accuracy),
            'historical_data': df['Close'].tolist(),
            'dates': df.index.strftime('%Y-%m-%d').tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
