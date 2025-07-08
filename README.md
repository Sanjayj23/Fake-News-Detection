# 📰 Fake News Detection Chatbot using BERT

This project is a **Streamlit chatbot** that classifies news text as **Real** ✅ or **Fake** ❌ using a fine-tuned **BERT transformer model**. It uses a Hugging Face-hosted model for live predictions and a simple interface for user input and result display.

---

## 📌 Features

* ✅ Fine-tuned BERT model for binary classification
* ✅ Detects fake news with confidence score
* ✅ Clean and intuitive Streamlit chatbot interface
* ✅ Deployed on Streamlit Cloud
* ✅ Model hosted on Hugging Face Hub

---

## 📋 Repository Structure

```
bert_fake_news_detector/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── secrets.toml        # (Optional) Hugging Face token (local testing)
└── README.md               # This file
```

---

## 📊 Model Details

* **Model**: `bert-base-uncased`
* **Fine-tuned** on: `true.csv` (real news) and `fake.csv` (fake news)
* **Hosted on**: [Hugging Face Model Hub](https://huggingface.co/shi13u/fake_news_detection_bert)
* **Classes**: `0 = Fake`, `1 = Real`

---

## 🧐 How It Works

1. Users input a news article or headline.
2. Text is tokenized using the model's tokenizer.
3. The fine-tuned BERT model generates logits.
4. The highest softmax score determines the prediction class.
5. Output is rendered as a label with a confidence percentage.

---

## 🚀 Running the App Locally

### 🔧 Requirements

```bash
pip install -r requirements.txt
```

### 🔑 Set Hugging Face Token (optional for private models)

Create a file `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "your_hf_token_here"
```

Then run:

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

🔗 [Try the app on Streamlit Cloud](https://fakenewsdetection-kzjxnzujdacfrze8pbplax.streamlit.app/)

---

## 📦 Hugging Face Model Repo

* 📁 [https://huggingface.co/shi13u/fake\_news\_detection\_bert](https://huggingface.co/shi13u/fake_news_detection_bert)

---

## 📺 Demo Video

📺 `demo.mp4` — A 1-minute walkthrough showing real-time fake/real predictions from the chatbot.

---

## 🧪 Example Inputs

| Input Text                                              | Prediction | Confidence |
| ------------------------------------------------------- | ---------- | ---------- |
| "NASA launches Artemis I mission to Moon"               | ✅ Real     | 70.2%      |
| "Bill Gates plans to microchip the world with vaccines" | ❌ Fake     | 60.8%      |

---

## 🛠️ Tools Used

* `transformers` by Hugging Face
* `BERT-base-uncased`
* `Streamlit`
* `torch` (PyTorch)

---

## 🤝 Acknowledgments

This project was developed as part of a final assignment for **fake news detection project by PPOC cell IIT Kanpur** to apply transformer-based models in real-world NLP tasks and showcase end-to-end deployment.
