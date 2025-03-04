from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_required, current_user
import joblib
from models import db, User, HeartDiseasePrediction, DiabetesPrediction, LungCancerPrediction
from auth import auth

# Konfiguracja dla różnych zbiorów danych - definiuje cechy i ich opisy dla każdego typu predykcji
DATASETS_CONFIG = {
    'heart_disease': {
        'features': [
            ('sex', 'Płeć pacjenta (0 = kobieta, 1 = mężczyzna)'),
            ('age', 'Wiek pacjenta'),
            ('cp', 'Typ bólu w klatce piersiowej'),
            ('trestbps', 'Ciśnienie tętnicze krwi w spoczynku'),
            ('chol', 'Poziom cholesterolu'),
            ('fbs', 'Cukier we krwi na czczo'),
            ('restecg', 'Wyniki EKG spoczynkowego'),
            ('thalachh', 'Maksymalne tętno'),
            ('exng', 'Dławica wysiłkowa'),
            ('oldpeak', 'Obniżenie odcinka ST'),
            ('slp', 'Nachylenie odcinka ST'),
            ('caa', 'Liczba głównych naczyń wieńcowych'),
            ('thall', 'Wynik testu Thallium')
        ]
    },
    'diabetes': {
        'features': [
            ('Pregnancies', 'Liczba ciąż'),
            ('Glucose', 'Poziom glukozy'),
            ('BloodPressure', 'Ciśnienie krwi'),
            ('SkinThickness', 'Grubość fałdu skórnego'),
            ('Insulin', 'Poziom insuliny'),
            ('BMI', 'Wskaźnik masy ciała'),
            ('DiabetesPedigreeFunction', 'Funkcja rodowodu cukrzycy'),
            ('Age', 'Wiek')
        ]
    },
    'lung_cancer': {
        'features': [
            ('GENDER', 'Płeć (M/F)'),
            ('AGE', 'Wiek'),
            ('SMOKING', 'Palenie tytoniu (1-2)'),
            ('YELLOW_FINGERS', 'Żółte palce (1-2)'),
            ('ANXIETY', 'Niepokój (1-2)'),
            ('PEER_PRESSURE', 'Presja rówieśników (1-2)'),
            ('CHRONIC DISEASE', 'Choroba przewlekła (1-2)'),
            ('FATIGUE', 'Zmęczenie (1-2)'),
            ('ALLERGY', 'Alergia (1-2)'),
            ('WHEEZING', 'Świszczący oddech (1-2)'),
            ('ALCOHOL CONSUMING', 'Spożywanie alkoholu (1-2)'),
            ('COUGHING', 'Kaszel (1-2)'),
            ('SHORTNESS OF BREATH', 'Duszność (1-2)'),
            ('SWALLOWING DIFFICULTY', 'Trudności w połykaniu (1-2)'),
            ('CHEST PAIN', 'Ból w klatce piersiowej (1-2)')
        ]
    }
}

# Inicjalizacja i konfiguracja aplikacji Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'

# Inicjalizacja bazy danych z użyciem skonfigurowanej aplikacji
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicjalizacja bazy danych z użyciem skonfigurowanej aplikacji
db.init_app(app)

# Konfiguracja systemu logowania
login_manager = LoginManager() # Utworzenie menedżera logowania
login_manager.init_app(app) # Powiązanie menedżera z aplikacją
login_manager.login_view = 'auth.login' # Ustawienie widoku logowania
login_manager.login_message = 'Proszę się zalogować.' # Komunikat dla niezalogowanych użytkowników

# Rejestracja blueprintu autoryzacji (mechanizm Flaska służący do organizacji funkcjonalności związanych z uwierzytelnianiem użytkowników)
app.register_blueprint(auth)

# Funkcja pomocnicza dla Flask-Login, ładująca użytkownika na podstawie ID
@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id)) # Pobranie użytkownika z bazy danych po ID

# Tworzenie wszystkich tabel w bazie danych
with app.app_context():
    db.create_all()

# Słownik przechowujący załadowane modele ML
loaded_models = {}

def load_models(dataset_name):
    """
    Ładuje modele uczenia maszynowego dla wybranego zbioru danych.
    Modele są ładowane tylko raz i przechowywane w pamięci.
    """
    if dataset_name not in loaded_models:
        models_dir = f'models/{dataset_name}'
        loaded_models[dataset_name] = {
            'rf': joblib.load(f'{models_dir}/rf_model.joblib'), # Random Forest
            'lr': joblib.load(f'{models_dir}/lr_model.joblib'), # Logistic Regression
            'dt': joblib.load(f'{models_dir}/dt_model.joblib'), # Decision Tree
            'scaler': joblib.load(f'{models_dir}/scaler.joblib') # Standaryzator danych
        }

def prepare_input_data(dataset_name, form_data):
    """
    Przygotowuje dane wejściowe do formatu akceptowanego przez modele.
    Konwertuje dane z formularza na odpowiedni format numeryczny.
    """
    features = DATASETS_CONFIG[dataset_name]['features']
    input_data = []

    for feature_name, _ in features:
        value = form_data[feature_name]
        if dataset_name == 'lung_cancer' and feature_name == 'GENDER':
            value = 1 if value.upper() == 'M' else 0
        input_data.append(float(value))

    return input_data

# Ścieżka dla strony głównej
@app.route('/')
@login_required
def index():
    """Strona główna - wyświetla listę dostępnych zbiorów danych"""
    return render_template('index.html', datasets=DATASETS_CONFIG.keys())

# Ścieżka dla formularza wprowadzania danych
@app.route('/dataset/<dataset_name>')
@login_required
def dataset_form(dataset_name):
    """
    Wyświetla forumlarz dla wybranego zbioru danych
    """
    if dataset_name not in DATASETS_CONFIG:
        return "Nieznany zbiór danych", 404

    features = DATASETS_CONFIG[dataset_name]['features']
    return render_template('dataset_form.html',
                           dataset_name=dataset_name,
                           features=features)

# Scieżka do usuwania predykcji
@app.route('/delete_prediction/<dataset_name>/<int:prediction_id>')
@login_required
def delete_prediction(dataset_name, prediction_id):
    """
    Usuwa wybraną predykcję z bazy danych
    """
    try:
        # Wybór odpowiedniego modelu w zależności od typu danych
        if dataset_name == 'heart_disease':
            prediction = HeartDiseasePrediction.query.get_or_404(prediction_id)
        elif dataset_name == 'diabetes':
            prediction = DiabetesPrediction.query.get_or_404(prediction_id)
        else:  # lung_cancer
            prediction = LungCancerPrediction.query.get_or_404(prediction_id)

        # Sprawdzenie uprawnień - tylko właściciel może usunąć predykcję
        if prediction.user_id != current_user.id:
            flash('Nie masz uprawnień do usunięcia tej predykcji.')
            return redirect(url_for('history', dataset_name=dataset_name))

        # Usunięcie predykcji
        db.session.delete(prediction)
        db.session.commit()
        return redirect(url_for('history', dataset_name=dataset_name))
    except Exception as e:
        return render_template('error.html', error=str(e))

#Ścieżka do historii predykcji
@app.route('/history/<dataset_name>')
@login_required
def history(dataset_name):
    """
    Wyświetla historię predykcji dla wybranego zbioru danych
    """

    try:
        # Pobranie predykcji z bazy danych w zależności od typu
        if dataset_name == 'heart_disease':
            predictions = HeartDiseasePrediction.query.filter_by(user_id=current_user.id).order_by(HeartDiseasePrediction.timestamp.desc()).all()
        elif dataset_name == 'diabetes':
            predictions = DiabetesPrediction.query.filter_by(user_id=current_user.id).order_by(DiabetesPrediction.timestamp.desc()).all()
        else:  # lung_cancer
            predictions = LungCancerPrediction.query.filter_by(user_id=current_user.id).order_by(LungCancerPrediction.timestamp.desc()).all()

        return render_template('history.html',
                               dataset_name=dataset_name,
                               predictions=predictions)
    except Exception as e:
        return render_template('error.html', error=str(e))

# Ścieżka do wykonywania finalnych predykcji
@app.route('/predict/<dataset_name>', methods=['POST'])
@login_required
def predict(dataset_name):
    """
    Główna funkcja wykonująca predykcje na podstawie wprowadzonych danych
    1. Sprawdza poprawność danych
    2. Ładuje odpowiednie modele
    3. Przygotowuje dane wejściowe
    4. Wykonuje predykcję wszystkimi modelami
    5. Zapisuje wyniki do bazy danych
    6. Zwraca wyniki użytkownikowi
    """
    try:
        # Sprawdzenie czy zbiór danych istnieje
        if dataset_name not in DATASETS_CONFIG:
            return "Nieznany zbiór danych", 404

        # Załadowanie modeli jeśli jeszcze nie są załadowane
        if dataset_name not in loaded_models:
            load_models(dataset_name)

        models = loaded_models[dataset_name]
        input_data = prepare_input_data(dataset_name, request.form)

        # Walidacja liczby cech
        expected_features = len(DATASETS_CONFIG[dataset_name]['features'])
        if len(input_data) != expected_features:
            raise ValueError(f"Nieprawidłowa liczba cech. Oczekiwano {expected_features}, otrzymano {len(input_data)}")

        # Skalowanie danych wejściowych
        input_scaled = models['scaler'].transform([input_data])

        # Wykonanie predykcji wszystkimi modelami
        predictions = {
            'Random Forest': {
                'prediction': int(models['rf'].predict(input_scaled)[0]),
                'probability': float(models['rf'].predict_proba(input_scaled)[0][1])
            },
            'Logistic Regression': {
                'prediction': int(models['lr'].predict(input_scaled)[0]),
                'probability': float(models['lr'].predict_proba(input_scaled)[0][1])
            },
            'Decision Tree': {
                'prediction': int(models['dt'].predict(input_scaled)[0]),
                'probability': float(models['dt'].predict_proba(input_scaled)[0][1])
            }
        }

        # Zapisz predykcje do bazy danych
        if dataset_name == 'heart_disease':
            prediction = HeartDiseasePrediction(
                user_id=current_user.id,
                sex=float(input_data[0]),
                age=float(input_data[1]),
                cp=float(input_data[2]),
                trestbps=float(input_data[3]),
                chol=float(input_data[4]),
                fbs=float(input_data[5]),
                restecg=float(input_data[6]),
                thalachh=float(input_data[7]),
                exng=float(input_data[8]),
                oldpeak=float(input_data[9]),
                slp=float(input_data[10]),
                caa=float(input_data[11]),
                thall=float(input_data[12]),
                rf_prediction=predictions['Random Forest']['prediction'],
                rf_probability=predictions['Random Forest']['probability'],
                lr_prediction=predictions['Logistic Regression']['prediction'],
                lr_probability=predictions['Logistic Regression']['probability'],
                dt_prediction=predictions['Decision Tree']['prediction'],
                dt_probability=predictions['Decision Tree']['probability']
            )
        elif dataset_name == 'diabetes':
            prediction = DiabetesPrediction(
                user_id=current_user.id,
                pregnancies=float(input_data[0]),
                glucose=float(input_data[1]),
                blood_pressure=float(input_data[2]),
                skin_thickness=float(input_data[3]),
                insulin=float(input_data[4]),
                bmi=float(input_data[5]),
                diabetes_pedigree_function=float(input_data[6]),
                age=float(input_data[7]),
                rf_prediction=predictions['Random Forest']['prediction'],
                rf_probability=predictions['Random Forest']['probability'],
                lr_prediction=predictions['Logistic Regression']['prediction'],
                lr_probability=predictions['Logistic Regression']['probability'],
                dt_prediction=predictions['Decision Tree']['prediction'],
                dt_probability=predictions['Decision Tree']['probability']
            )
        else:  # lung_cancer
            prediction = LungCancerPrediction(
                user_id=current_user.id,
                gender=request.form['GENDER'],
                age=float(input_data[1]),
                smoking=float(input_data[2]),
                yellow_fingers=float(input_data[3]),
                anxiety=float(input_data[4]),
                peer_pressure=float(input_data[5]),
                chronic_disease=float(input_data[6]),
                fatigue=float(input_data[7]),
                allergy=float(input_data[8]),
                wheezing=float(input_data[9]),
                alcohol_consuming=float(input_data[10]),
                coughing=float(input_data[11]),
                shortness_of_breath=float(input_data[12]),
                swallowing_difficulty=float(input_data[13]),
                chest_pain=float(input_data[14]),
                rf_prediction=predictions['Random Forest']['prediction'],
                rf_probability=predictions['Random Forest']['probability'],
                lr_prediction=predictions['Logistic Regression']['prediction'],
                lr_probability=predictions['Logistic Regression']['probability'],
                dt_prediction=predictions['Decision Tree']['prediction'],
                dt_probability=predictions['Decision Tree']['probability']
            )

        # Zapisanie predykcji do bazy danych
        db.session.add(prediction)
        db.session.commit()

        # Zwrócenie wyników
        return render_template('result.html',
                               dataset_name=dataset_name,
                               predictions=predictions)

    except Exception as e:
        return render_template('error.html', error=str(e))

# Uruchomienie aplikacji w trybie debug
if __name__ == '__main__':
    app.run(debug=True)