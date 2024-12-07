# Investigating the Impact of Prompt Design on Energy Efficiency of Large Language Models (LLMs)

![Project Banner](assets/banner2.png)

## 📜 Project Overview
This project analyzes how different prompt designs influence the energy efficiency of Large Language Models (LLMs) during inference. The study evaluates resource utilization, power consumption, and performance across various NLP tasks (e.g., classification, sentiment analysis, question answering) using state-of-the-art models such as:

- [**Qwen2.5-7B**](https://huggingface.co/Qwen/Qwen2.5-7B)
- [**Gemma-2-2b-it**](https://huggingface.co/google/gemma-2-2b-it)
- [**Mistral-7B-Instruct-v0.3**](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)


By understanding the relationship between prompt type and energy efficiency, this research aims to provide actionable insights for sustainable AI deployment.

---

## 🚀 Key Features
- **Performance Analysis:** Comprehensive evaluation of inference time, CPU/GPU utilization, VRAM usage, and power consumption.
- **Energy Consumption Insights:** Explore the energy usage of different prompt types across models.
- **Statistical Analysis:** Includes ANOVA, correlation, and normality testing to validate findings.
- **Visualization:** Rich visualizations (e.g., bar charts, line plots) for a clear understanding of results.

---

## 📊 Research Questions
1. How do different prompt types (classification, sentiment analysis, question answering) affect model performance and energy consumption?
2. What are the resource utilization patterns across prompt types and models?
3. How do inference efficiency and computational requirements vary between prompt types?

---

## 📂 Repository Structure

├── data/                     # Dataset used for experiments
├── scripts/                  # Python and R scripts for analysis
├── figures/                  # Generated plots and visualizations
├── results/                  # Final processed results
├── README.md                 # Project documentation (this file)
└── main.tex                  # LaTeX source for project report
