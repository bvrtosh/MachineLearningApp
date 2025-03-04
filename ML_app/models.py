from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Inicjalizacja obiektu bazy danych
db = SQLAlchemy()

class User(UserMixin, db.Model):
    """
    Model użytkownika systemu przechowujący podstawowe informacje oraz relacje z predykcjami.
    Atrybuty:
       id: Unikalny identyfikator użytkownika
       username: Nazwa użytkownika (max 80 znaków)
       email: Email użytkownika (max 120 znaków)
       password_hash: Zahashowane hasło

   Relacje:
       heart_disease_predictions: Relacja z predykcjami chorób serca
       diabetes_predictions: Relacja z predykcjami cukrzycy
       lung_cancer_predictions: Relacja z predykcjami raka płuc

   Dziedziczy po UserMixin aby zapewnić integrację z Flask-Login.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    heart_disease_predictions = db.relationship('HeartDiseasePrediction', backref='user', lazy=True)
    diabetes_predictions = db.relationship('DiabetesPrediction', backref='user', lazy=True)
    lung_cancer_predictions = db.relationship('LungCancerPrediction', backref='user', lazy=True)

    def set_password(self, password):
        """
        Hashuje i zapisuje hasło użytkownika
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """
        Weryfikuje hasło użytkownika
        """
        return check_password_hash(self.password_hash, password)

class HeartDiseasePrediction(db.Model):
    """
    Model przechowujący predykcje chorób serca.
    Zawiera zarówno dane wejściowe jak i wyniki predykcji.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Cechy wejściowe
    sex = db.Column(db.Integer)
    age = db.Column(db.Integer)
    cp = db.Column(db.Integer)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    fbs = db.Column(db.Integer)
    restecg = db.Column(db.Integer)
    thalachh = db.Column(db.Float)
    exng = db.Column(db.Integer)
    oldpeak = db.Column(db.Float)
    slp = db.Column(db.Integer)
    caa = db.Column(db.Integer)
    thall = db.Column(db.Integer)

    # Predykcje
    rf_prediction = db.Column(db.Integer)
    rf_probability = db.Column(db.Float)
    lr_prediction = db.Column(db.Integer)
    lr_probability = db.Column(db.Float)
    dt_prediction = db.Column(db.Integer)
    dt_probability = db.Column(db.Float)

class DiabetesPrediction(db.Model):
    """
    Model przechowujący predykcje cukrzycy.
    Zawiera zarówno dane wejściowe jak i wyniki predykcji.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Cechy wejściowe
    pregnancies = db.Column(db.Integer)
    glucose = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    diabetes_pedigree_function = db.Column(db.Float)
    age = db.Column(db.Integer)

    # Predykcje
    rf_prediction = db.Column(db.Integer)
    rf_probability = db.Column(db.Float)
    lr_prediction = db.Column(db.Integer)
    lr_probability = db.Column(db.Float)
    dt_prediction = db.Column(db.Integer)
    dt_probability = db.Column(db.Float)

class LungCancerPrediction(db.Model):
    """
    Model przechowujący predykcje raka płuc.
    Zawiera zarówno dane wejściowe jak i wyniki predykcji.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Cechy wejściowe
    gender = db.Column(db.String(1))
    age = db.Column(db.Integer)
    smoking = db.Column(db.Integer)
    yellow_fingers = db.Column(db.Integer)
    anxiety = db.Column(db.Integer)
    peer_pressure = db.Column(db.Integer)
    chronic_disease = db.Column(db.Integer)
    fatigue = db.Column(db.Integer)
    allergy = db.Column(db.Integer)
    wheezing = db.Column(db.Integer)
    alcohol_consuming = db.Column(db.Integer)
    coughing = db.Column(db.Integer)
    shortness_of_breath = db.Column(db.Integer)
    swallowing_difficulty = db.Column(db.Integer)
    chest_pain = db.Column(db.Integer)

    # Predykcje
    rf_prediction = db.Column(db.Integer)
    rf_probability = db.Column(db.Float)
    lr_prediction = db.Column(db.Integer)
    lr_probability = db.Column(db.Float)
    dt_prediction = db.Column(db.Integer)
    dt_probability = db.Column(db.Float)