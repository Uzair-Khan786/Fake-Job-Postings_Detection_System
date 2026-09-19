# 🔍 Fake Job Postings Detection

**An AI-powered system for detecting fraudulent job postings using Machine Learning, Deep Learning, and BERT.**


## 📌 Overview

**Online job platforms have made recruitment easier, but they have also created opportunities for fraudulent job postings and employment scams.**

**This project develops an AI-based classification system to distinguish between legitimate and potentially fraudulent job postings by analyzing textual and structured information from job advertisements.**

## 🚀 Three different approaches are explored:

### 🤖 Version A: Traditional Machine Learning Models

### 🧠 Version B: Deep Learning using BiLSTM

### 🤗 Version C: BERT Fine-Tuning

**The models are evaluated on a dataset containing approximately 16,000 job postings.**

## 🎯 Objective

**The primary objective is to develop an intelligent system capable of detecting potentially fraudulent job advertisements using NLP and machine learning techniques.**

**The project focuses on:**

**Cleaning and preprocessing job-posting data.**

**Extracting meaningful textual features.**

**Applying ML and deep learning classification techniques.**

**Fine-tuning BERT for fake-job classification.**

**Comparing different approaches using standard evaluation metrics.**

## 🧠 Project Workflow

                    ┌─────────────────────┐
                    │   Kaggle Dataset    │
                    │   ~16K Job Posts    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ Cleaning & NLP      │
                    └──────────┬──────────┘
                               ↓
              ┌────────────────┼────────────────┐
              ↓                ↓                ↓
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │ ML Models   │  │   BiLSTM    │  │    BERT     │
       │ Version A   │  │ Version B   │  │ Version C   │
       └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
              └────────────────┼────────────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ Accuracy / F1 etc.  │
                    └─────────────────────┘


## 📂 Dataset

**Source: Kaggle**

**Dataset Size: ~16,000 job postings**

### The dataset contains information related to job advertisements, including:

**Job title**

**Company profile**

**Job description**

**Requirements**

**Benefits**

**Location**

**Employment information**

**Fraudulent/legitimate label**

## 🔬 Approaches

### 🤖 Version A — Machine Learning

**Traditional machine learning models are trained using engineered features extracted from the job-posting data.**

**Pipeline:**

**Job Text → Preprocessing → Feature Extraction → ML Model → Classification**

### 🧠 Version B — BiLSTM/BiGRU/ANN

**Compared Bidirectional Long Short-Term Memory (BiLSTM) , Bidirectional Gated Units (BiGRU) and Artificial Neural Networks to capture sequential and contextual patterns within job-posting text.**

**Text → Tokenization → Embedding → BiLSTM → Dense Layer → Classification**

### 🤗 Version C — BERT Fine-Tuning

**A pretrained BERT model is fine-tuned on the job-posting dataset to learn contextual representations for fraudulent-job classification.**

**Text → BERT Tokenizer → BERT → Fine-Tuning → Classification**

## 📊 Model Performance

### The three approaches were evaluated on the ~16K-row dataset.


| Version | Approach / Model | Accuracy | Precision | Recall | F1-Score |
|:-------:|:-----------------|:--------:|:---------:|:------:|:--------:|
| **🟦 Version A** | Machine Learning (SVC) | **~99%** | **~93%** | **~92%** | **~92%** |
| **🟩 Version B** | Deep Learning (BiLSTM) | **~99%** | **~93%** | **~91%** | **~92%** |
| **🟪 Version C** | BERT Fine-Tuning | **~98%** | **~93%** | **~76%** | **~84%** |

**The ML and BiLSTM approaches achieved approximately 99% accuracy with balanced precision, recall, and F1-scores. BERT achieved approximately 98% accuracy and 93% precision, while its recall was comparatively lower on this dataset.**

## 📊 Exploratory Data Analysis

**The project includes visual analysis of the dataset, including:**

**Class distribution**

**Fraudulent vs. legitimate job postings**

**Word clouds of fake job texts**

**Feature distributions**

**Text-based analysis**

**Model performance comparison**

## 🛠️ Tech Stack

Technology	Purpose

🐍 Python	Programming

📓 Jupyter Notebook	Development & experimentation

🐼 Pandas	Data manipulation

🔢 NumPy	Numerical computation

📊 Matplotlib / Seaborn	Visualization

☁️ WordCloud	Text visualization

🤖 Scikit-learn	Machine Learning

🧠 PyTorch	Deep Learning

🔄 BiLSTM	Sequential text classification

🤗 Transformers	BERT fine-tuning

📝 NLP	Text preprocessing


## 🚀 Getting Started

### Clone the Repository

git clone <your-repository-url>

cd Fake-Job-Postings-Detection

### Install Dependencies

pip install -r requirements.txt

### Run the Notebook

jupyter notebook

Open Fake_Job_Postings_Detection.ipynb and run the cells sequentially.


## 👨‍💻 Author

**Uzair Ahmad Khan - Data Science Enthusiast**

## ⭐ If you found this project interesting, consider starring the repository!
