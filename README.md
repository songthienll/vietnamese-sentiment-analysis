# Vietnamese Food Review Sentiment Analysis

🍜 **Vietnamese Food Review Sentiment Analysis** is a web application built with Streamlit that analyzes the sentiment of user-provided reviews about food. The project leverages a fine-tuned **PhoBERT** model to classify sentiments as either **Tích cực (Positive)** or **Tiêu cực (Negative)**, providing insights into user feedback with confidence scores. The application is designed to process Vietnamese text, incorporating preprocessing steps such as URL removal, stopword elimination, and tokenization using the `pyvi` library.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [License](#license)

## Project Overview
This project aims to classify sentiments expressed in Vietnamese food reviews using a fine-tuned PhoBERT model. The application is built using Streamlit for an interactive web interface, allowing users to input reviews and receive sentiment predictions along with confidence scores. The preprocessing pipeline includes text cleaning, Vietnamese tokenization, and stopword removal to ensure accurate sentiment analysis.

## Features
- **Sentiment Classification**: Classifies reviews as Positive or Negative using a fine-tuned PhoBERT model.
- **Confidence Scores**: Provides a confidence score for each prediction.
- **Interactive UI**: Streamlit-based interface for easy input and visualization of results.
- **Vietnamese Text Processing**: Utilizes `pyvi` for tokenization and a custom stopword list for preprocessing.

## Demo
Explore the live application here: [Vietnamese Sentiment Analysis](https://vietnamese-sentiment-analysis.streamlit.app/)

Below is a screenshot of the Streamlit app showing a sample sentiment analysis result:

![Demo Screenshot](https://raw.githubusercontent.com/songthienll/vietnamese-sentiment-analysis/main/assets/demo.png)


## Installation
To run the application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/songthienll/vietnamese-sentiment-analysis.git
   cd vietnamese-sentiment-analysis
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have a `requirements.txt` file with the following dependencies:
   ```
   streamlit
   torch
   transformers
   pyvi
   ```

4. **Download Stopwords File**:
   Ensure the `vietnamese-stopwords.txt` file is present in the project directory, as it is required for text preprocessing.

5. **Download Model**:
   The application uses a fine-tuned PhoBERT model from `songthienll/phobert-model`. Ensure internet access for automatic model downloading via Hugging Face's `transformers` library.

## Usage
1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   This will launch the web application in your default browser.

2. **Enter a Review**:
   Input a Vietnamese food review in the text area (e.g., "Phở này ngon tuyệt!" or "Cơm tấm hơi khô.").

3. **Analyze Sentiment**:
   Click the "Phân tích" button to get the sentiment prediction and confidence score.

4. **View Results**:
   - Positive sentiment (😊) is displayed with a green message.
   - Negative sentiment (😞) is displayed with a red message.
   - The confidence score is shown as a percentage.

## Model Details
- **Model**: PhoBERT (`songthienll/phobert-model`), fine-tuned for sentiment analysis on Vietnamese food reviews.
- **Training Dataset**: The model was trained on a dataset of 50,000 reviews, split into training (30,000), validation (10,000), and test sets (10,000), with balanced labels (50% Positive, 50% Negative).
- **Preprocessing**:
  - Text cleaning: Converts to lowercase, removes URLs, special characters, and extra spaces.
  - Tokenization: Uses `pyvi.ViTokenizer` for Vietnamese word segmentation.
  - Stopword Removal: Filters out common Vietnamese stopwords using `vietnamese-stopwords.txt`.
- **Performance**: Achieves approximately 90% accuracy, with precision, recall, and F1-score of 0.90 for both classes (based on the test set evaluation).

## Dataset
The dataset used for fine-tuning is sourced from `/kaggle/input/vietnamese-food-review-dataset/`, containing:
- `train.csv`: 30,000 reviews.
- `valid.csv`: 10,000 reviews.
- `test.csv`: 10,000 reviews.
Each review is labeled as `0` (Negative) or `1` (Positive).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
