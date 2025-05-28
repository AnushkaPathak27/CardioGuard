from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from models import db, User, Prediction
import pickle
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cardioguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

def get_reset_token(email):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    return serializer.dumps(email, salt='password-reset-salt')

def verify_reset_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=expiration)
        return email
    except:
        return None

def load_model(model_path='classifier.pkl'):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

model = load_model()

FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
VALIDATION_RULES = {
    'age': (20, 100), 'sex': (0, 1), 'cp': (0, 3), 'trestbps': (80, 200),
    'chol': (100, 600), 'fbs': (0, 1), 'restecg': (0, 2), 'thalach': (60, 220),
    'exang': (0, 1), 'oldpeak': (0, 10), 'slope': (0, 2), 'ca': (0, 3), 'thal': (1, 3)
}

def validate_input(data):
    errors = {}
    for field, (min_val, max_val) in VALIDATION_RULES.items():
        try:
            if field not in data:
                errors[field] = f"{field} is required"
                continue
            value = float(data[field])
            if not min_val <= value <= max_val:
                errors[field] = f"Must be between {min_val} and {max_val}"
        except ValueError:
            errors[field] = "Must be a number"
    return errors

def prepare_features(data):
    features = [float(data[field]) for field in FEATURE_NAMES]
    return pd.DataFrame([features], columns=FEATURE_NAMES)

@app.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('questionnaire'))
    return render_template('homepg.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('questionnaire'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('questionnaire'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('questionnaire'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('questionnaire'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.")
    return redirect(url_for('landing'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            token = get_reset_token(email)
            reset_url = url_for('reset_password', token=token, _external=True)
            
            return render_template('forgot_password.html', reset_link=reset_url)
        flash('Email address not found.', 'error')
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = verify_reset_token(token)
    if email is None:
        flash('Invalid or expired reset token', 'error')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        new_password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
            db.session.commit()
            flash('Password updated! Log in now.', 'success')
            return redirect(url_for('login'))
    return render_template('reset_password.html')

@app.route('/index.html')
@login_required
def questionnaire():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        errors = validate_input(data)
        if errors:
            return jsonify({"errors": errors}), 400
        X = prepare_features(data)
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        new_prediction = Prediction(
            user_id=current_user.id,
            **{key: data[key] for key in FEATURE_NAMES},
            result='DISEASE DETECTED' if prediction == 1 else 'NO DISEASE DETECTED',
            probability=prediction_proba[1] if prediction == 1 else prediction_proba[0]
        )
        db.session.add(new_prediction)
        db.session.commit()
        return jsonify({
            "result": "DISEASE DETECTED" if prediction == 1 else "NO DISEASE DETECTED",
            "probability": prediction_proba[1] if prediction == 1 else prediction_proba[0],
            "redirect": "/heart-disease-info" if prediction == 1 else "/heart-disease-nil"
        })
    except Exception as e:
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/heart-disease-info')
@login_required
def disease_info():
    return render_template('heart-disease-info.html')

@app.route('/heart-disease-nil')
@login_required
def disease_nil():
    return render_template('heart-disease-nil.html')

@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)


@app.route('/delete_history', methods=['POST'])
@login_required
def delete_history():
    try:
        # Get prediction ID from request
        prediction_id = request.form.get('prediction_id')
        
        if prediction_id == 'all':
            # Delete all predictions for current user
            Prediction.query.filter_by(user_id=current_user.id).delete()
        else:
            # Delete specific prediction
            prediction = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first()
            if prediction:
                db.session.delete(prediction)
            else:
                return jsonify({'error': 'Prediction not found'}), 404
        
        db.session.commit()
        flash('History deleted successfully', 'success')
        return redirect(url_for('history'))
        
    except Exception as e:
        db.session.rollback()
        flash('Error deleting history', 'error')
        return redirect(url_for('history'))
    
if __name__ == '__main__':
    app.run(debug=True)
