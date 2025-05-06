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

🤝 Conclusions
reveals that certain topics attract more biased language than others. 'Trump presidency' and 'white nationalism' have the highest average bias rates, with over 70% of sentences labeled as biased. These findings suggest that politically and socially charged topics are more prone to emotionally loaded language. On the other hand, topics like 'immigration', 'universal healthercare', and 'gun control' show lower bias rates, though they still remain near 45-50%, indicating that even these texts are not free from bias.

Understanding which topics are more likely to contain biased language helps contextualize the limitations of people-contributed platforms like Wikipedia and can inform targeted moderation or review efforts

sentiment polarity across biased and unbiased text of different articles. It reveals that while both categories generally maintain fa neutral tone on average, biased articles tend to exhibit a wider range of sentiment. This includes a higher frequency of strongly positive or negative sentiment, indicating that biased writing often uses more emotionally charge language. In contrast, unbiased texts shows a tighter sentiment distribution, reflecting a more balanced and objective tone. This suggests that sentiment polarity can serve as a useful signal in detecting bias, particularly when combined with other linguistic and contextual features.

Top biased words are claim (claims, claimed, claiming), illegal, aliens.

data suggests that media outlets vary widely in how frequently they use biased language. Alternet and Federalist have the highest average bias scores, suggesting strong ideological framing. Breitbart and HuffPost also show high bias, while MSNBC and Fox News despite opposing views - have similar moderate scores.Reuters and USA Today stand out for their low bias, reflecting more neutral reporting. Overall, outlets with a strong ideological orientation tend to have higher bias scores, whereas mainstream or centrist outlets tend to score lower. In particular, Reuters stands out for having the most neutral tone in its reporting.


📄 License
MIT License – see LICENSE file for details.

🙌 Acknowledgments
Thanks to Wikipedia, open-source libraries, and the Code Jam organizers for providing this challenge!
