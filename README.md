# 🧠 A7_NLP-Assignment: Distillation vs LoRA

This project is part of the **AT82.05 - Artificial Intelligence: Natural Language Understanding (NLU)** course assignment. It compares **Distillation (Odd vs Even Layer Student Training)** and **LoRA (Low-Rank Adaptation)** for toxic comment classification using BERT-based models. It also includes a **Streamlit web application** for real-time prediction.

---

## 📌 Project Structure

```
A7_NLP-Assignment/
├── A7.ipynb                   # Main Jupyter Notebook (Tasks 1–4)
├── webapp.py                 # Streamlit app (Task 5)
├── student_lora/            # LoRA adapter folder
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── results/                  # Evaluation screenshots
│   └── Screenshot_*.png
├── README.md                # This file
```

---

## 📊 Tasks Overview

| Task | Description |
|------|-------------|
| ✅ Task 1 | Load and prepare a hate speech / toxic comment dataset |
| ✅ Task 2 | Train a 6-layer student using odd `{1,3,5,7,9,11}` and even `{2,4,6,8,10,12}` teacher layers |
| ✅ Task 3 | Fine-tune the student model using **LoRA** |
| ✅ Task 4 | Evaluate and compare all models on test data |
| ✅ Task 5 | Build a **web application** to classify input as toxic or not |

---

## 📥 Dataset

We use the [Hate Speech and Offensive Language Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language), which contains over 24k tweets labeled as hate speech, offensive language, or neither. Labels are converted into binary: `1 = toxic`, `0 = not toxic`.

---

## 🧠 Model Types

| Model      | Method         | Layers Used                     |
|------------|----------------|----------------------------------|
| Odd Layer  | Distillation   | Teacher layers `{1,3,5,7,9,11}` |
| Even Layer | Distillation   | Teacher layers `{2,4,6,8,10,12}`|
| LoRA       | Adapter-based  | Full 12-layer model w/ LoRA     |

---

## 📈 Results

| Model       | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|-----------|--------|----------|
| Odd Layer   | ...      | ...       | ...    | ...      |
| Even Layer  | ...      | ...       | ...    | ...      |
| **LoRA**    | ...      | ...       | ...    | ...      |

📷 See the [`results/`](results/) folder for screenshots of predictions and evaluation logs.

---

## 🌐 Web App (Task 5)

A minimal **Streamlit-based web app** to classify text as **toxic** or **not toxic** using the fine-tuned **LoRA model**.

### ▶️ Run the app locally:

```bash
pip install -r requirements.txt
streamlit run webapp.py
```

### 📷 Screenshot

![Toxic Comment Prediction](results/Screenshot%202025-03-22%20133025.png)

---

## 🧪 Test Examples

Try these in the app:

- `"I hate you!"` → ⚠️ **Toxic**
- `"You're awesome, have a nice day!"` → ✅ **Not Toxic**

---

## ⚙️ Requirements

```text
torch>=2.1.0
transformers>=4.36.2
peft>=0.7.1
streamlit
scikit-learn
```

Install them via:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

- 👨‍🎓 Student ID: **PK_124960**
- 📅 Assignment Due: March 2025
- 📘 Course: AT82.05 - Artificial Intelligence: NLU

---

## 📚 References

- [Davidson et al., 2017 Hate Speech Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT (LoRA Library)](https://github.com/huggingface/peft)

---

> 📌 *This assignment was completed as part of an academic course and is not intended for production use.*
