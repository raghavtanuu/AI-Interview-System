# 🤖 AI Interview System

## 📌 Overview

The **AI Interview System** is a full-stack backend application that simulates a real interview environment. It evaluates candidate responses using AI-based scoring techniques such as **TF-IDF** and **semantic similarity**.

---

## 🚀 Features

* 🔐 **JWT Authentication** (Candidate & Recruiter)
* 🧠 **AI-Based Answer Evaluation**

  * TF-IDF Similarity
  * Semantic Similarity (Transformers)
* 🎤 **Interview Simulation**

  * 15 Random Questions
  * 60-second timer per question
* 📊 **Performance Analysis**

  * Question-wise scoring
  * Final selection decision
* 👨‍💼 **Recruiter Dashboard**

  * View candidate performance
  * Detailed answer analysis
* 🎥 **Integrity System**

  * Tab switch detection
  * Camera & mic verification

---

## 🛠️ Tech Stack

* **Backend:** FastAPI
* **Database:** SQLite / MySQL
* **ORM:** SQLAlchemy
* **Authentication:** JWT (PyJWT)
* **AI/ML:**

  * Scikit-learn (TF-IDF)
  * Sentence Transformers
* **Frontend:** HTML + CSS + JavaScript

---

## 📂 Project Structure

```bash
.
├── tanu.py              # Main FastAPI backend
├── ai_interview.db      # Database (auto-generated)
├── assets/              # Images (optional)
├── .gitignore
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/raghavtanuu/AI-Interview-System.git
cd AI-Interview-System
```

### 2️⃣ Install Dependencies

```bash
pip install fastapi uvicorn sqlalchemy pydantic passlib[bcrypt] PyJWT scikit-learn sentence-transformers
```

### 3️⃣ Run Server

```bash
uvicorn tanu:app --reload
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:8000
```

---

## 🔑 API Endpoints

| Endpoint                    | Method | Description      |
| --------------------------- | ------ | ---------------- |
| `/auth/register`            | POST   | Register user    |
| `/auth/login`               | POST   | Login            |
| `/interviews`               | POST   | Create interview |
| `/interviews/{id}/start`    | POST   | Start interview  |
| `/interviews/{id}/answers`  | POST   | Submit answer    |
| `/interviews/{id}/complete` | POST   | Finish interview |

---

## 📊 AI Scoring Logic

The system evaluates answers using:

* **TF-IDF Similarity** → Keyword matching
* **Semantic Similarity** → Meaning-based comparison
* **Keyword Overlap** → Concept matching
* **Final Score** → Weighted combination

---

## 🧠 Future Improvements

* Add **voice emotion detection**
* Deploy using cloud (AWS / Render)
* Add **real-time cheating detection AI**
* Improve UI with React

---

## 📚 References

* FastAPI – https://fastapi.tiangolo.com/
* SQLAlchemy – https://www.sqlalchemy.org/
* Scikit-learn – https://scikit-learn.org/
* Sentence Transformers – https://www.sbert.net/

---

## 👨‍💻 Author

**Tanu Raghav**
Mathematics Department
Jaypee Institute of Information and Technology

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
