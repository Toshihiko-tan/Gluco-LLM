# Gluco‑LLM  
**Evaluating the Zero‑Shot Predictive Ability of Large Language Models for Continuous Glucose Monitoring Data**  

This repository contains the code, poster, and completed honors thesis for “Evaluating the Zero‑Shot Predictive Ability of Large Language Models for Continuous Glucose Monitoring Data,” submitted in partial fulfillment of the requirements for Honors in Data Science at the University of Michigan (W25).  

A poster presenting preliminary results won **Best Poster** at the 2025 Michigan Student Symposium for Interdisciplinary Statistical Sciences (MSSISS 2025).

---

## Table of Contents
- [Overview](#overview)  
- [Credits](#credits)  
- [Dependencies](#dependencies)  
- [Installation](#installation)  
- [Repository Structure](#repository-structure)  
- [Configuration & Hyperparameters](#configuration--hyperparameters)  
- [Usage Examples](#usage-examples)  
  - [Training Baseline Models](#training-baseline-models)  
  - [Hyperparameter Tuning](#hyperparameter-tuning)  
  - [Running LLM Methods](#running-llm-methods)  
- [Data](#data)  
- [License](#license)  

---

## Overview  
We explore zero‑shot forecasting of CGM time series using two paradigms:  
1. **Direct‑Prompt** — format recent readings + context as a text prompt to a pre‑trained LLM.  
2. **Retrieval‑Augmented Generation (RAG)** — prepend retrieved historical segments to the prompt.  

We benchmark against classical baselines (ARIMA, Linear Regression), a Transformer, and Latent ODE.  

---

## Credits  
- **Advisor:** Prof. Irina Gaynanova (Biostatistics, UMich)  
- **Mentor:** Renat Sergazinov, PhD (Meta; GlucoBench project lead)  
- **GlucoBench** (data formatting, preprocessing, baseline implementations):  
  https://github.com/IrinaStatsLab/GlucoBench  

---

## Dependencies  
- Python 3.10  
- See `requirements.txt` for Python packages  
- **torchdiffeq** (for Latent ODE)  
- **OpenAI** & **DeepSeek** API keys (for LLM methods)  
  - Store keys in a `.env` file and load via `python-dotenv`

---

## Installation  

```bash
# Clone this repo
git clone https://github.com/yourusername/Gluco‑LLM.git
cd Gluco‑LLM

# Install Python deps
pip install -r requirements.txt

# Install torchdiffeq
pip install torchdiffeq
