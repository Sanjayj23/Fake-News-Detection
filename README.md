# Fake News Detection with BERT

An end-to-end NLP project that fine-tunes BERT for binary fake/real news classification, hosts the trained model on Hugging Face, and serves predictions through a Streamlit application.

## Reliability improvements

- Loads and caches the model across Streamlit reruns.
- Uses model label metadata instead of blindly assuming label order.
- Validates empty, very short, and excessively long input.
- Reports truncation at BERT's 512-token limit.
- Marks confidence below 65% as uncertain and shows both class probabilities.
- Handles download and prediction failures with useful messages.
- Includes nine offline unit tests for validation, labels, and probability interpretation.
- Distinguishes pattern classification from evidence-based fact checking.

## Repository structure

| File | Purpose |
| --- | --- |
| `Fake_News_Detection_BERT.ipynb` | BERT fine-tuning and evaluation workflow |
| `Upload_model_on_HF.ipynb` | Model and tokenizer upload to Hugging Face |
| `detector.py` | Reusable inference, validation, and confidence logic |
| `app.py` | Streamlit user interface |
| `tests/test_detector.py` | Offline unit tests |

## Run locally

```bash
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run app.py
```

The default checkpoint is `shi13u/fake_news_detection_bert`. Set `HF_MODEL_NAME` to use another compatible sequence-classification model. Set `HF_TOKEN` for a private checkpoint.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Live demo

[Open the Streamlit application](https://fake-news-detection-bert.streamlit.app/)

## Limitations

The training notebook currently samples 100 real and 100 fake articles, so this is a learning prototype rather than a production fact-checker. Its saved outputs do not include a final held-out metric; this repository therefore makes no unsupported accuracy claim. Real-world use would require a larger representative dataset, drift monitoring, calibration, bias analysis, and independent fact verification.

## Technology

Python, PyTorch, Transformers, Hugging Face Hub, BERT, Streamlit, and unittest.

## Author

[Sanjay Jangir](https://github.com/Sanjayj23)
