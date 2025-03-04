# Software Description

## a. Short Name  
**Model Comparator**  

## b. Full Name  
**Comparative Analysis of Selected Machine Learning Models**  

## c. Short Description  
The application utilizes three different medical datasets to support disease diagnosis. After the user provides answers to specific questions, the program analyzes the information and presents the probability of a given disease (**lung cancer, diabetes, or heart disease**).  

The results are calculated based on three selected machine learning models:  
- **Logistic Regression**  
- **Decision Tree**  
- **Random Forest**  

---

# Copyright

## a. Authors  
- **Adam Wrzałek**  
- **Bartosz Deptuła**  
- **Mikołaj Mazur**  

## b. Licensing Terms  

**MIT License Copyright (c) 2025 Adam Wrzałek, Bartosz Deptuła, Mikołaj Mazur**  

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.  

---

# Installation Guide

## Requirements
- **PyCharm Community Edition** (latest available version)

## Environment Setup (After launching PyCharm)
1. **File > New Project**  
   - Location: `Your_Project_Name`  
   - Click **Create**

2. **Install Required Packages**  
   Run the following command in the terminal:
   ```sh
   pip install flask pandas numpy scikit-learn joblib flask_sqlalchemy datetime flask-login
   ```
   Wait until the command prompt reappears.

3. **(GitHub) Download the Application**  
   - Click **"<> Code" > Download ZIP**
   - Place the application folder inside your project directory
   - Extract the ZIP file (using WinRAR or built-in extractor)

4. **Run the Application**  
   - Double-click `flask-app.py` in the `ML_app` folder
   - Click the **Run** icon in the top-right corner
   - In the "Run" tab, click the link next to **"Running on _"**
   - You can now use the application!

# Requirements Specification  

| ID  | Name | Description | Priority | Category |
|------|----------------|---------------------------------|-----------|------------|
| I01  | Decision Tree Model  | Splits available data into smaller subsets based on selected features. | Required | Functional |
| I02  | Random Forest Model | Creates multiple decision trees on random subsets of data and combines their results. | Required | Functional |
| I03  | Logistic Regression Model | Allows modeling of probability for event occurrence. | Required | Functional |
| I04  | Application | Built with Flask framework, responsible for data processing and presenting model results. | Required | Functional |
| I05  | Data | Selection of three datasets related to medical issues. | Required | Functional |
| I07  | Graphical User Interface | Intuitive UI with data visualization and model results. | Useful | Functional |
| I08  | Data Comparison | Enables model comparison across different datasets. | Useful | Functional |
| I09  | Saving Results | Saves calculated probabilities for each sample. | Useful | Functional |
| I10  | Registration & Login | Allows user account registration and authentication. | Optional | Functional |
| I11  | K-Means Clustering Model | Divides dataset into k clusters using centroids. | Optional | Functional |

---

# System Architecture  

The application follows a **client-server architecture** with three main components:  
- **Frontend** (HTML, CSS) – Provides user interaction via forms, prediction results, and analysis history.  
- **Backend** (Flask Framework) – Handles user data processing, application logic, and communication with machine learning models.  
- **Machine Learning Models** (Random Forest, Logistic Regression, Decision Tree) – Dynamically loaded based on the selected dataset (e.g., heart disease, diabetes, lung cancer).  

The application processes user input, scales data, generates predictions, and stores results in a database using **SQLAlchemy**. The architecture supports:  
- **Filtering results** (e.g., displaying history by dataset)  
- **Managing predictions** (e.g., deleting selected analyses)  
- **Presenting results** in a user-friendly manner  

Due to its modular structure, the system can be easily extended with new predictive models or additional functionalities.  

---

# Libraries Used  

| Library | Version | Description | Functions Used |
|---------|---------|--------------------------------|----------------------|
| Flask | 3.1.0 | Web application framework | Flask, render_template, request, redirect, url_for, flash, Blueprint |
| Flask-Login | 0.6.3 | User session & authentication management | LoginManager, login_required, current_user, login_user, logout_user, UserMixin |
| Flask-SQLAlchemy | 3.1.1 | Database configuration for Flask | SQLAlchemy |
| DateTime | 5.5 | Date & time handling | datetime |
| joblib | 1.2.0 | Loading machine learning models | joblib.load, joblib.dump |
| Werkzeug | 3.1.3 | Password hashing | generate_password_hash, check_password_hash |
| scikit-learn | 1.3.0 | Classification, regression, data splitting, model evaluation | RandomForestClassifier, DecisionTreeClassifier, LogisticRegression, train_test_split, StandardScaler, classification_report |
| Pandas | 2.0.3 | Data loading from CSV files | pd.read_csv |

