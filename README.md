# 🏏 IPL Match Winner Prediction

An end-to-end machine learning project that predicts the winner of an IPL cricket match
based on teams, venue, city, and toss information. Built with a full data pipeline —
from raw, messy match data to a deployed interactive prediction app.

## 🚀 Live Demo
```bash
streamlit run app.py
```
*(Add a screenshot or a deployed Streamlit Cloud link here once hosted — this is a strong resume touch.)*

## 📌 Project Highlights
- **1,000+ match records** processed through a full preprocessing pipeline
- **Data cleaning**: handled missing values, standardized inconsistent team names
  (e.g. `MI`, `Mumbai Indians`, `mumbai indians` → `Mumbai Indians`)
- **Feature engineering**: toss-winner alignment, toss-decision encoding, venue/city encoding
- **Model comparison**: Logistic Regression, Decision Tree, Random Forest, and XGBoost,
  evaluated on Accuracy, Confusion Matrix, and Classification Report
- **Deployment-ready**: best model auto-selected and served via a Streamlit web app

## 🗂️ Project Structure
```
ipl-match-winner-prediction/
├── data/
│   ├── generate_data.py      # synthetic dataset generator (swap for real Kaggle IPL data)
│   ├── ipl_matches.csv       # raw match data
│   └── ipl_matches_clean.csv # cleaned + feature-engineered output
├── src/
│   ├── preprocessing.py      # missing values, team name standardization, feature engineering
│   └── train_models.py       # trains & compares 4 models, saves the best one
├── models/
│   ├── best_model.pkl        # serialized best-performing model
│   ├── encoders.pkl          # label encoders used at inference time
│   └── results.json          # accuracy + confusion matrix per model
├── app.py                    # Streamlit deployment app
├── requirements.txt
└── README.md
```

## 🛠️ Setup & Usage

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/<your-username>/ipl-match-winner-prediction.git
   cd ipl-match-winner-prediction
   pip install -r requirements.txt
   ```

2. **Generate the dataset**
   *(or replace `data/ipl_matches.csv` with the real [Kaggle IPL dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-matches-20082020) using the same column names)*
   ```bash
   python data/generate_data.py
   ```

3. **Train and compare models**
   ```bash
   cd src
   python train_models.py
   ```
   This prints Accuracy / Confusion Matrix / Classification Report for all four models
   and saves the best-performing one to `models/`.

4. **Run the deployment app**
   ```bash
   cd ..
   streamlit run app.py
   ```

## 📊 Model Comparison

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~0.51    |
| Decision Tree       | ~0.51    |
| Random Forest       | ~0.54    |
| **XGBoost**          | **~0.58**    |

*(Exact numbers vary by run/dataset — see `models/results.json` after training.
Note: results here are on a synthetic dataset; expect higher, more meaningful
accuracy on the real historical IPL dataset, where team form, head-to-head
history, and player-level stats provide much stronger signal.)*

## 🔮 Future Improvements
- Incorporate player-level features (batting/bowling form, head-to-head stats)
- Add recent-form / rolling win-rate features instead of static team strength
- Hyperparameter tuning via GridSearchCV / Optuna
- Track experiments with MLflow
- Deploy publicly on Streamlit Community Cloud

## 📄 License
MIT
