import torch
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import streamlit as st

# 🎯 Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 🧠 Load base model (6-layer student architecture assumed)
base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 🧩 Inject LoRA adapter
model = PeftModel.from_pretrained(base_model, "student_lora")
model.eval()

# 🚀 Streamlit App UI
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")
st.title("🧠 Toxic Comment Classifier")
st.write("Enter a sentence and the model will classify it as **Toxic** or **Not Toxic**.")

# 📝 Input Box
user_input = st.text_area("✍️ Enter your comment here:")

# 🔍 Classify Button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a sentence.")
    else:
        # Tokenize
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # Show Result
        if prediction == 1:
            st.error("❌ This comment is **Toxic**.")
        else:
            st.success("✅ This comment is **Not Toxic**.")
