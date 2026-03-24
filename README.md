# 🧠 AI-Based Answer Script Evaluation System

An intelligent system that automatically evaluates handwritten answer scripts using OCR and Natural Language Processing (NLP). This project converts handwritten answers into structured text and evaluates them against a reference answer using semantic similarity and keyword analysis.

---

## 🚀 Features

* 📸 Handwritten Answer Recognition (OCR)
* 🧹 Text Preprocessing & Spell Correction
* 🧠 Semantic Similarity Evaluation
* 🔑 Keyword Matching System
* 📊 Automated Scoring Mechanism
* 🌐 REST API using FastAPI

---

## 🏗️ System Architecture

### 🔹 Stage 1: Image Processing & Text Extraction

* Input: Scanned/photographed answer sheet
* Image preprocessing (grayscale, denoising, cropping)
* Text detection using PaddleOCR
* Text recognition using TrOCR

### 🔹 Stage 2: Text Preprocessing

* Spell correction
* Grammar normalization
* Structuring raw OCR output into readable format

### 🔹 Stage 3: Answer Evaluation

* Semantic similarity comparison with reference answer
* Keyword-based scoring
* Final score generation

---

## 🧰 Tech Stack

* **Backend:** Python, FastAPI
* **OCR:** PaddleOCR, TrOCR (Transformers)
* **NLP:** Sentence Transformers, NLTK / SpaCy
* **Deep Learning:** PyTorch
* **Image Processing:** OpenCV

---

## 📁 Project Structure

```
ai-answer-evaluation/
│
├── app/
│   ├── app.py              # FastAPI entry point
│   ├── routes/             # API routes
│   ├── services/           # OCR, NLP logic
│   ├── models/             # ML model loading
│   └── utils/              # Helper functions
│
├── data/                   # Sample images / test data
├── notebooks/              # Experiments
├── Requirement.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/shrejan/Ai-Answer-Evaluation-System.git
cd ai-answer-evaluation
```

### 2. Create virtual environment

```
python -m venv venv
```

### 3. Activate environment

```
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 4. Install dependencies

```
pip install -r Requirement.txt
```

---

## ▶️ Running the Application

```
uvicorn app.main:app --reload
```

Open in browser:

* http://127.0.0.1:8000
* API Docs: http://127.0.0.1:8000/docs

---

## 📡 API Endpoint

### 🔹 Extract & Evaluate Answer

**POST** `/orc`

#### Request:

* Upload image file (handwritten answer sheet)

#### Response:

```
{
  "extracted_text": "...",
  "processed_text": "...",
  "similarity_score": 0.87,
  "keyword_score": 8,
  "final_score": 85
}
```

---

## 🧪 Future Improvements

* 📈 Model fine-tuning for better handwriting recognition
* 🌍 Multi-language support
* 📊 Teacher dashboard with analytics
* 🧾 PDF answer sheet support
* ⚡ Batch processing API

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* Microsoft TrOCR
* PaddleOCR
* Hugging Face Transformers

---

## 📬 Contact

For any queries or collaboration:
📧 [shrejankotyan.com]

---

