# 🧠 Wikipedia Bias Detection

This project was created for a **Code Jam challenge** focused on assessing potential **bias in Wikipedia articles**. 
Wikipedia strives for neutrality, but since it is publicly editable, implicit bias can still creep into the content. 
This project applies data science and machine learning techniques to detect such bias, ultimately producing a "bias score" for new articles.

---

## 📌 Objective

Build a machine learning pipeline that:
1. **Collects and preprocesses** Wikipedia article data.
2. **Analyzes** the linguistic and sentiment-based patterns in the text.
3. **Trains** a supervised learning model to classify biased vs. unbiased sentences.
4. **Predicts** the bias of new Wikipedia articles via a custom scoring function.

---

## 📂 Project Structure

```bash
├── data/                     # Raw and processed datasets
├── notebooks/                # EDA and experimentation notebooks
├── src/                      # Scripts for data processing, modeling, and prediction
├── models/                   # Trained models
├── outputs/                  # Evaluation results and charts
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
└── bias_predictor.py         # Script to run bias prediction on new Wikipedia articles
🔄 Workflow
1. Data Collection
Data is sourced from:

Wikipedia API

Potential third-party datasets

Custom web scraping (if needed)

2. Exploratory Data Analysis (EDA)
We investigate:

Term frequency (TF-IDF)

Sentiment distribution

Use of subjective/adjective-heavy language

Named entity usage and patterns

3. Supervised Learning
We classify bias at the sentence level using:

Logistic Regression

Random Forest

BERT-based transformer models (optional for advanced versions)

4. Bias Scoring Function
A function that:

Predicts sentence-level bias

Aggregates results into a normalized bias score for the article

🚀 How to Use
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/wiki-bias-detector.git
cd wiki-bias-detector
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run a prediction

bash
Copy
Edit
python bias_predictor.py --url "https://en.wikipedia.org/wiki/Some_Article"
Output

Bias score printed to console

Optionally, sentence-level bias predictions

📊 Visuals
<p align="center"> <img src="notebooks/figures/sentiment_distribution.png" width="400"/> <img src="notebooks/figures/bias_score_example.png" width="400"/> </p>
🛠️ Tech Stack
Python 3.x

Scikit-learn

NLTK / spaCy

HuggingFace Transformers (optional)

Wikipedia-API

Jupyter for EDA

✅ To-Do
 Data collection from Wikipedia API

 EDA of linguistic features

 Sentence-level bias classification

 Bias scoring pipeline

 Model improvement with more labeled data

 Deploy as a web app or API

🤝 Contributors


📄 License
MIT License – see LICENSE file for details.

🙌 Acknowledgments
Thanks to Wikipedia, open-source libraries, and the Code Jam organizers for providing this challenge!
