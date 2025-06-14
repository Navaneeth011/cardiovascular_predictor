# 🫀 Cardiovascular Disease Risk Predictor

A comprehensive machine learning web application that predicts cardiovascular disease risk using patient health data. Built with Flask, scikit-learn, and modern responsive design, this application provides accurate risk assessment with detailed analytics and historical tracking capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Output Images](#output-images)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## 🎯 Overview

Cardiovascular disease remains one of the leading causes of mortality worldwide. Early detection and risk assessment are crucial for prevention and timely intervention. This project leverages machine learning algorithms to analyze various health parameters and provide personalized cardiovascular risk predictions.

The application uses a Random Forest classifier trained on a comprehensive cardiovascular dataset containing over 70,000 medical records. The model considers multiple risk factors including age, gender, blood pressure, cholesterol levels, lifestyle habits, and physical measurements to generate accurate risk assessments.

The web interface provides an intuitive user experience with real-time form validation, comprehensive result visualization, and historical trend analysis. Healthcare professionals and individuals can use this tool to understand cardiovascular risk factors and track health metrics over time.

## ✨ Features

### Core Functionality
- **Intelligent Risk Assessment**: Advanced machine learning model predicting cardiovascular disease risk with high accuracy
- **Real-time Validation**: Comprehensive input validation with instant feedback for medical parameters
- **Personalized Results**: Detailed risk analysis with specific risk factors identification
- **BMI Calculation**: Automatic body mass index calculation and interpretation
- **Blood Pressure Analysis**: Systolic and diastolic pressure evaluation with medical recommendations

### Advanced Analytics
- **Historical Tracking**: Complete prediction history with trend analysis and comparison charts
- **Risk Trend Visualization**: Interactive charts showing cardiovascular risk changes over time
- **Health Metrics Comparison**: Personal health metrics compared against dataset averages
- **Risk Factor Analysis**: Detailed breakdown of contributing risk factors with explanations
- **Statistical Insights**: Comprehensive health analytics with actionable recommendations

### User Experience
- **Responsive Design**: Modern, mobile-friendly interface that works across all devices
- **Interactive Charts**: Dynamic visualizations using Chart.js for better data understanding
- **Form Memory**: Smart form data retention to prevent data loss during validation errors
- **Medical Guidance**: Clear explanations of medical terms and risk factors
- **Accessibility Features**: Screen reader compatible with proper semantic markup

## 🛠 Technology Stack

### Backend Technologies
- **Python 3.8+**: Core programming language for machine learning and web development
- **Flask**: Lightweight web framework for rapid development and deployment
- **scikit-learn**: Machine learning library for model training and prediction
- **pandas**: Data manipulation and analysis for dataset processing
- **NumPy**: Numerical computing for efficient mathematical operations
- **pickle**: Model serialization for persistent storage and loading

### Frontend Technologies
- **HTML5**: Modern semantic markup for accessibility and SEO
- **Tailwind CSS**: Utility-first CSS framework for responsive design
- **JavaScript (ES6+)**: Interactive functionality and form validation
- **Chart.js**: Professional data visualization and charting library
- **Responsive Design**: Mobile-first approach with cross-browser compatibility

### Data Storage
- **JSON**: Lightweight data storage for prediction history
- **CSV**: Dataset format for model training and validation
- **Pickle**: Binary format for trained model persistence

## 📊 Dataset Information

The application is trained on the Cardiovascular Disease Dataset, which contains comprehensive medical records with the following characteristics:

### Dataset Specifications
- **Total Records**: 70,000+ patient records
- **Features**: 11 input features plus target variable
- **Data Quality**: Cleaned and preprocessed with outlier removal
- **Age Range**: 18-100 years (converted from days to years)
- **Gender Distribution**: Balanced representation of male and female patients

### Feature Descriptions
- **Age**: Patient age in years (18-100)
- **Gender**: Biological sex (1: Female, 2: Male)
- **Height**: Body height in centimeters (100-250 cm)
- **Weight**: Body weight in kilograms (30-200 kg)
- **Systolic BP**: Systolic blood pressure in mmHg (60-200)
- **Diastolic BP**: Diastolic blood pressure in mmHg (40-120)
- **Cholesterol**: Cholesterol level (1: Normal, 2: Above normal, 3: Well above normal)
- **Glucose**: Glucose level (1: Normal, 2: Above normal, 3: Well above normal)
- **Smoking**: Smoking status (0: No, 1: Yes)
- **Alcohol**: Alcohol consumption (0: No, 1: Yes)
- **Physical Activity**: Regular physical activity (0: No, 1: Yes)

### Data Preprocessing
The dataset undergoes rigorous cleaning and preprocessing to ensure model accuracy. This includes removing unrealistic medical values, handling outliers, normalizing numerical features, and ensuring data consistency. The preprocessing pipeline includes blood pressure validation (systolic > diastolic), BMI range verification, and age boundary enforcement.

## 🚀 Installation

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Navaneeth011/cardiovascular_predictor.git
   cd cardiovascular-disease-predictor
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset**
   Download the cardiovascular dataset (`cardio_train.csv`) and place it in the project root directory. The dataset should be in semicolon-separated format.

5. **Train the Model** (Optional)
   ```bash
   python model.py
   ```
   This step is optional as the application will automatically train the model if no pre-trained model is found.

6. **Run the Application**
   ```bash
   python app.py
   ```

7. **Access the Application**
   Open your web browser and navigate to `http://localhost:5000`

### Requirements.txt
```
Flask==2.3.3
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
Werkzeug==2.3.7
Jinja2==3.1.2
MarkupSafe==2.1.3
click==8.1.7
itsdangerous==2.1.2
```

## 💻 Usage

### Making Predictions

1. **Access the Main Interface**: Navigate to the application homepage where you'll find the comprehensive health information form.

2. **Fill Personal Information**: Enter your age, gender, and physical measurements (height and weight). The application automatically validates input ranges.

3. **Enter Medical Data**: Provide blood pressure readings (systolic and diastolic), cholesterol levels, and glucose levels using the dropdown menus.

4. **Lifestyle Information**: Select your smoking status, alcohol consumption habits, and physical activity level.

5. **Submit for Analysis**: Click the "Analyze Cardiovascular Risk" button to receive your personalized risk assessment.

### Understanding Results

The prediction results include several key components:

- **Risk Level**: Clear classification as "High Risk" or "Low Risk"
- **Probability Score**: Numerical percentage indicating the likelihood of cardiovascular disease
- **Risk Factors**: Detailed list of identified risk factors contributing to your assessment
- **Health Profile**: Summary of your key health metrics including BMI and blood pressure classification
- **Recommendations**: General guidance based on identified risk factors

### Historical Analysis

Access your prediction history through the "View Prediction History" button to see:

- **Trend Analysis**: Visual charts showing how your cardiovascular risk changes over time
- **Metric Comparisons**: Your health metrics compared against dataset averages
- **Detailed History**: Tabular view of all previous predictions with timestamps
- **Statistical Insights**: Analysis of risk factor patterns and health improvements

## 📈 Model Performance

### Algorithm Selection
The application uses a Random Forest classifier, chosen for its robustness, interpretability, and excellent performance on medical datasets. Random Forest provides several advantages for cardiovascular risk prediction:

- **Feature Importance**: Clear ranking of which health factors most influence risk
- **Handling Non-linear Relationships**: Captures complex interactions between health variables
- **Robustness**: Resistant to overfitting and handles missing data well
- **Interpretability**: Provides insights into decision-making process

### Performance Metrics
- **Accuracy**: Approximately 73-75% on test dataset
- **Cross-validation Score**: 5-fold CV with consistent performance
- **Precision**: High precision for high-risk predictions to minimize false positives
- **Recall**: Balanced recall to ensure most at-risk patients are identified
- **F1-Score**: Optimized for medical screening applications

### Feature Importance
The model identifies the following as the most important predictive features:
1. **Age**: Strong predictor with risk increasing significantly after 55
2. **Systolic Blood Pressure**: Primary cardiovascular risk indicator
3. **Weight/BMI**: Strong correlation with cardiovascular complications
4. **Cholesterol Levels**: Significant impact on arterial health
5. **Diastolic Blood Pressure**: Important for overall cardiovascular assessment

### Model Validation
The model undergoes rigorous validation including:
- **Train-Test Split**: 80-20 split with stratified sampling
- **Cross-Validation**: 5-fold cross-validation for robust performance estimation
- **Medical Validation**: Results aligned with established medical guidelines
- **Continuous Monitoring**: Model performance tracked with new predictions

## 🖼 Output Images

### Main Prediction Interface
![Main Interface](![alt text](image.png))
*The main application interface featuring the comprehensive health information form with real-time validation and modern responsive design.*

### Prediction Results Display
![Prediction Results](![alt text](image-2.png))
*Detailed prediction results showing risk level, probability score, identified risk factors, and personalized health profile summary.*

### Historical Analytics Dashboard
![Analytics Dashboard](![alt text](image-3.png))
*Comprehensive analytics dashboard with interactive charts showing cardiovascular risk trends and health metrics comparison over time.*

## 🔗 API Endpoints

### Core Endpoints

#### `GET /`
- **Description**: Main application homepage with prediction form
- **Response**: HTML page with health information form
- **Features**: Form validation, responsive design, medical guidance

#### `POST /predict`
- **Description**: Process health data and return cardiovascular risk prediction
- **Request Body**: Form data with health parameters
- **Response**: Prediction results with detailed analysis
- **Validation**: Comprehensive input validation with medical constraints

#### `GET /history`
- **Description**: Display prediction history with analytics dashboard
- **Response**: Historical data with interactive charts and trends
- **Features**: Risk trend analysis, health metrics comparison

#### `GET /api/predictions`
- **Description**: RESTful API endpoint for prediction data
- **Response**: JSON array of all stored predictions
- **Usage**: Integration with external applications or data analysis

#### `GET /clear_history`
- **Description**: Clear all stored prediction history
- **Response**: Redirect to history page
- **Security**: Includes confirmation dialog for data protection

### API Response Format

```json
{
  "prediction": 1,
  "probability": 0.75,
  "risk_level": "High Risk",
  "risk_factors": [
    "High systolic blood pressure",
    "Elevated cholesterol levels",
    "Lack of physical activity"
  ],
  "bmi": 28.5,
  "timestamp": "2024-01-15T10:30:00"
}
```

## 📁 Project Structure

```
cardiovascular_predictor/
├── app.py                 # Flask application (main backend)
├── model.py              # ML model training & preprocessing
├── templates/
│   ├── index.html        # User input form & results
│   └── history.html      # Analytics dashboard
├── requirements.txt      # Dependencies
├── cardio_train.csv     # Dataset (user provides)
├── cardio_model.pkl     # Trained model (auto-generated)
└── predictions.json     # History storage (auto-generated)
```

### File Descriptions

- **app.py**: Main Flask application handling web routes, form processing, and prediction logic
- **model.py**: Machine learning implementation including data preprocessing, model training, and prediction functions
- **templates/**: Jinja2 HTML templates with responsive design and interactive features
- **cardio_model.pkl**: Serialized trained model with scaler and metadata
- **predictions.json**: JSON storage for prediction history and analytics
- **requirements.txt**: Python package dependencies for easy installation

## 🤝 Contributing

We welcome contributions to improve the Cardiovascular Disease Risk Predictor! Here's how you can contribute:

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Set up development environment following installation instructions
4. Make your changes with appropriate tests
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Contribution Guidelines
- **Code Quality**: Follow PEP 8 style guidelines for Python code
- **Documentation**: Update README and add docstrings for new functions
- **Testing**: Include unit tests for new features
- **Medical Accuracy**: Ensure medical information is accurate and cite sources
- **User Experience**: Maintain accessibility and responsive design standards

### Areas for Contribution
- **Model Improvements**: Enhanced algorithms, feature engineering, hyperparameter tuning
- **User Interface**: Improved design, accessibility features, internationalization
- **Analytics**: Advanced visualization, statistical analysis, reporting features
- **Documentation**: API documentation, user guides, medical explanations
- **Testing**: Unit tests, integration tests, performance testing

## 📄 License

This project is licensed under the License process
- ❌ No liability or warranty
- ⚠️ License and copyright notice required

## ⚠️ Disclaimer

**Important Medical Disclaimer**: This application is designed for educational and informational purposes only. It is not intended as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

### Key Points
- **Not a Medical Device**: This tool is not FDA approved and should not be used for clinical decision-making
- **Educational Purpose**: Designed to help understand cardiovascular risk factors and promote health awareness
- **Consult Healthcare Providers**: Any concerns about cardiovascular health should be discussed with qualified medical professionals
- **No Warranty**: The predictions are based on statistical models and may not reflect individual medical circumstances
- **Data Privacy**: Users are responsible for protecting their personal health information

### Recommended Use
- **Health Education**: Understanding cardiovascular risk factors and their relationships
- **Risk Awareness**: General awareness of potential health risks based on lifestyle factors
- **Health Tracking**: Monitoring changes in health metrics over time
- **Medical Discussions**: Preparation for healthcare provider consultations
- **Research**: Academic and research applications in cardiovascular health

---

**For questions, support, or medical inquiries, please consult with qualified healthcare professionals. This application is a tool for education and awareness, not medical diagnosis or treatment.**