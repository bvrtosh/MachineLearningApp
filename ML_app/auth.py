from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required
from models import User, db

# Utworzenie blueprintu dla funkcjonalności autoryzacji
auth = Blueprint('auth', __name__)

# Ścieżka do logowania
@auth.route('/login', methods=['GET', 'POST'])
def login():
    """
    Obsługa logowania użytkownika.
    GET: Wyświetla formularz logowania
    POST: Weryfikuje dane i loguje użytkownika
    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        # Wyszukanie użytkownika w bazie
        user = User.query.filter_by(username=username).first()

        # Sprawdzenie wprowadzonych danych
        if not user or not user.check_password(password):
            flash('Sprawdź swoje dane logowania i spróbuj ponownie.')
            return redirect(url_for('auth.login'))

        # Zalogowanie użytkownika
        login_user(user, remember=remember)
        return redirect(url_for('index'))

    return render_template('login.html')

# Ścieżka do rejestracji
@auth.route('/register', methods=['GET', 'POST'])
def register():
    """
    Obsługa rejestracji nowego użytkownika.
    GET: Wyświetla formularz rejestracji
    POST: Tworzy nowego użytkownika jeśli dane są poprawne
    """
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Sprawdzenie czy użytkownik już istnieje
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Nazwa użytkownika już istnieje')
            return redirect(url_for('auth.register'))

        # Sprawdzenie czy email już istnieje
        email_exists = User.query.filter_by(email=email).first()
        if email_exists:
            flash('Email już istnieje')
            return redirect(url_for('auth.register'))

        # Utworzenie nowego użytkownika
        new_user = User(username=username, email=email)
        new_user.set_password(password)

        # Zapisanie użytkownika w bazie
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('auth.login'))

    return render_template('register.html')

# Ścieżka do wylogowania
@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))