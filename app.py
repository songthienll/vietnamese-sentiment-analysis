import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi.ViTokenizer import ViTokenizer
import re
import os

st.set_page_config(page_title="Phân tích cảm xúc món ăn", page_icon="🍜", layout="wide")

# Load model và tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    repo_id = 'songthienll/phobert-model'

    try:
        #Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo_id)

        #Load model
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        model.eval()

        return tokenizer, model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Tải stopwords và hàm loại bỏ stopwords
@st.cache_data
def load_stopwords():
    try:
        with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
            vietnamese_stopwords = set(line.strip() for line in f)
        return vietnamese_stopwords
    except FileNotFoundError:
        st.warning("File vietnamese-stopwords.txt does not exist.")
        return set()

# Load model, tokenizer và stopwords
tokenizer, model = load_model_and_tokenizer()
vietnamese_stopwords = load_stopwords()

# Định nghĩa label
id2label = {0: "Tiêu cực", 1: "Tích cực"}

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s\u00C0-\u1EF9]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    if not vietnamese_stopwords:
        return text
    return ' '.join([word for word in text.split() if word not in vietnamese_stopwords])

# Hàm tokenize văn bản
def tokenizer_text(text, tokenized=True, lowercased=True):
    text = preprocess_text(text)
    text = ViTokenizer.tokenize(text) if tokenized else text
    text = remove_stopwords(text)
    return text

# Hàm dự đoán sentiment
def predict_sentiment(text):
    if tokenizer is None or model is None:
        return "Error loading model", 0.0

    try:
        # Tiền xử lý văn bản
        processed_text = tokenizer_text(text)
        # Tokenize
        inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        # Dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Sử dụng detach() để tránh lỗi meta tensor
            logits = logits.detach()

            # Dự đoán class
            predicted_class = torch.argmax(logits, dim=1).item()

            # Tính xác suất
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0][predicted_class].item()

        return id2label[predicted_class], confidence

    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        return "Lỗi", 0.0

# Giao diện Streamlit
st.markdown("<h1 style='text-align: center; white-space: nowrap;'>🍜 Phân tích cảm xúc đánh giá món ăn</h1>",unsafe_allow_html=True)
st.markdown("Nhập đánh giá về món ăn để phân tích cảm xúc")
st.markdown("---")

# Kiểm tra xem model đã load thành công chưa
if tokenizer is None or model is None:
    st.error("Không thể load model. Vui lòng kiểm tra đường dẫn và file model.")
    st.stop()

user_input = st.text_area("Đánh giá của bạn:", height=150, placeholder="Ví dụ: 'Phở này ngon tuyệt!'")

if st.button("Phân tích", key="analyze_button"):
    if user_input:
        with st.spinner("Đang phân tích..."):
            sentiment, confidence = predict_sentiment(user_input)

            if sentiment == "Tích cực":
                st.success(f"**Kết quả Sentiment:** {sentiment} 😊")
                st.info(f"**Độ tin cậy:** {confidence:.2%}")
            elif sentiment == "Tiêu cực":
                st.error(f"**Kết quả Sentiment:** {sentiment} 😞")
                st.info(f"**Độ tin cậy:** {confidence:.2%}")
            else:
                st.error(f"**Kết quả:** {sentiment}")
    else:
        st.warning("Vui lòng nhập đánh giá trước khi phân tích!")

# Sidebar
st.sidebar.header("Thông tin ứng dụng")
st.sidebar.markdown("""
Ứng dụng này sử dụng model **PhoBERT** để phân tích cảm xúc của các đánh giá về món ăn Việt Nam.  
Hãy thử nhập một câu như:  
- "Bánh mì này quá ngon!"  
- "Cơm tấm hơi khô."  
""")
st.sidebar.image("https://raw.githubusercontent.com/songthienll/vietnamese-sentiment-analysis/main/assets/food.jpeg", use_container_width=True)
