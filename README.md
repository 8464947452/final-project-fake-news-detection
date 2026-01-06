# final-project-fake-news-detection
# Classifying Fake News Articles Using Machine Learning and Natural Language Processing (NLP)

This project aims to apply **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to classify fake news articles. The project compares traditional machine learning models (such as **TF-IDF + Logistic Regression**) with advanced transformer-based models (like **BERT**) to determine which approach provides higher accuracy and interpretability. The project also utilizes **Explainability** methods such as **SHAP** to offer insights into the decision-making process of the models.

## **Research Questions**
1. Can ML and NLP approaches reliably distinguish fake from real news articles across different domains?
2. Which linguistic and semantic cues most strongly indicate misinformation when analyzed using explainability methods?
3. How does model performance vary between shallow ML techniques and deep transformer models?

## **Dataset Details**
- **Name:** LIAR Dataset – Fake News Detection
- **Source:** University of California, Santa Barbara – William Yang Wang (2017)
- **Origin:** Collected from PolitiFact (2007–2016)
- **Details:** Contains over 12,000 short political statements, each manually labeled for truthfulness from “pants-on-fire” to “true.”

## **Libraries Used**
- **pandas** for data manipulation and preprocessing.
- **matplotlib** and **seaborn** for data visualization.
- **scikit-learn** for machine learning models and evaluation.
- **transformers** (BERT) for deep learning models.
- **SHAP** for model interpretability and feature importance analysis.

## **Steps Performed in the Notebook**
1. **Import Libraries:** All necessary libraries for data manipulation, model building, and evaluation are imported.
2. **Dataset Overview:** The dataset is loaded and previewed to understand its structure.
3. **Data Preprocessing:**
   - Clean and preprocess text data by removing stopwords, punctuation, and non-relevant characters.
   - Tokenize and prepare the text for machine learning and deep learning models.
4. **Feature Engineering:**
   - Generate **TF-IDF (Term Frequency-Inverse Document Frequency)** features for machine learning models.
   - Use **BERT** and other transformer-based models for deep learning-based feature extraction.
5. **Model Training:**
   - Train models like **Logistic Regression** using TF-IDF features.
   - Train **BERT** for text classification to detect fake news.
6. **Hyperparameter Tuning:** Fine-tune hyperparameters for both models to achieve optimal performance.
7. **Model Evaluation:** Evaluate the models using **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.
8. **Explainability (SHAP Analysis):** Use **SHAP** to understand the most influential features and explain the decision-making process of the models.
9. **Comparison of Models:** Compare the performance of shallow models (e.g., TF-IDF + Logistic Regression) with deep transformer models (e.g., BERT).

## **How to Run the Notebook**
1. Clone the repository to your local machine.
2. Install the required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn transformers shap
   ```
3. Download the **LIAR Dataset** or use your own dataset of political statements.
4. Open the notebook in **Jupyter Notebook** or **Google Colab**.
5. Execute all the cells to perform data preprocessing, model training, and evaluation.

## **Results**
This project compares the effectiveness of traditional machine learning models and transformer-based models for detecting fake news. **BERT** and other transformer models outperform traditional models in terms of accuracy and interpretability. **SHAP analysis** provides valuable insights into the key features contributing to the classification of fake news.

## **Contributions**
Feel free to contribute by:
- Adding more machine learning algorithms for fake news classification.
- Enhancing the text preprocessing and feature engineering techniques.
- Improving the model explainability and fairness.

