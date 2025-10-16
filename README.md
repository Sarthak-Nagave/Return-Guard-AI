# ReturnGuardAI

**ReturnGuardAI** is an AI-powered system designed to detect and prevent e-commerce return abuse. It uses machine learning to predict potentially fraudulent return requests, helping e-commerce businesses reduce losses and streamline their return management process.

---

## 🔹 Features
- Predict fraudulent return requests using ML models.
- Admin dashboard to monitor return patterns and risk scores.
- CSV logging of all transactions and flagged returns.
- Integration-ready with e-commerce platforms.
- Risk scoring and case escalation for high-risk returns.

---

## 🔹 Project Structure

```

ReturnGuardAI/
│
├── app.py                # Main Flask app
├── ml_pipeline.py        # ML pipeline for training & prediction
├── models.py             # Database models & ML constants
├── utils.py              # Helper functions (logging, scoring, etc.)
├── requirements.txt      # Python dependencies
├── README.md             # Project description
├── .gitignore            # Files/folders to ignore
├── config/               # Config files (e.g., DB config)
├── data/                 # Dataset folder (sample_data.csv)
├── templates/            # HTML templates
├── static/               # CSS, JS, images
└── logs/                 # Logs (ignored by git)

````

---

## 🔹 Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/ReturnGuardAI.git
cd ReturnGuardAI
````

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set up environment variables in `.env` if using database credentials or API keys.

---

## 🔹 Usage

1. Run the Flask app:

```bash
python app.py
```

2. Open your browser and go to:

```
http://127.0.0.1:5000
```

3. Use the web interface to submit return requests and view predictions.

---

## 🔹 Machine Learning Pipeline

* Uses **Random Forest** or **Logistic Regression** for classification.
* Features include return reason, user history, purchase data, and timestamps.
* Risk score calculated to prioritize high-risk returns.
* Escalation logs created for admin review.

---

## 🔹 Logging

* All requests are logged in CSV and optional database.
* High-risk cases are logged separately for escalation.
* Logs help track patterns and improve model performance.

---

## 🔹 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -m "Description"`)
4. Push the branch (`git push origin feature-branch`)
5. Create a Pull Request on GitHub

---

## 🔹 Author - 
Sarthak Nagave

## 🔹 License

This project is open-source under the MIT License. See [LICENSE](LICENSE) for details.


