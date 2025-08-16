# Evaluating LLMs for Historic Texts

This repository contains the code and datasets for investigating the ability of **Large Language Models (LLMs)** to understand and translate **Medieval Irish texts**.

The project evaluates both proprietary and local LLMs using custom-built benchmark datasets, exploring zero-shot and three-shot prompting with varying hyperparameters such as temperature, top-p, top-k, and quantization settings.

## Repository Structure

```
FYP_LLM_historic_texts/
│
├── benchmark_datasets/     # Benchmark datasets (eDIL, CELT, ISO, synthetic)
├── cross_model_evaluation/      # code for cross modal evaluation
├── outputs_inital_proprietary_models/  #csv files of outputs from chat-interface of proprietary models
├── threeshot_code/         # code for evaluation of proprietary models for three-shot prompting
├── web_scraper/            # code used to perform web scraping and create benchmark datasets
├── zeroshot_code/         # code for evaluation of all models for zero-shot prompting  
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Outline

* **Models Evaluated**:

  * Proprietary: GPT, Gemini, Grok
  * Open-source: LLaMA, Qwen, Phi, Gemma

* **Prompting Strategies**:

  * Few-shot 
  * Zero-shot 

* **Parameters Tested**:

  * Temperature: {0.1, 0.2, 0.5, 1.0}
  * Top-p: {0.7, 0.85, 0.9}
  * Top-k: {30, 40} *(not supported by GPT/Grok)*
  * Configurations tested:

    1. (0.1, 0.7, 30)
    2. (0.5, 0.85, 30)
    3. (1.0, 0.9, 40)
    4. (0.2, 0.85, 40)

* **Evaluation Metrics**:

  * BLEU
  * METEOR
  * BERTScore
  * SentenceBert
  * WordNet
  * Human annotation

---

## Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mamthamv/FYP_LLM_historic_texts.git
cd FYP_LLM_historic_texts
pip install -r requirements.txt
```

### Running Experiments

#### Local Models (Zero-Shot)

```bash
cd zeroshot_code/local_models
python main_file.py
```

* `config.py` can be modified to edit the models, prompt and benchmarks as required.

#### API-Based Models (Zero-Shot or Three-Shot)

```bash
cd zeroshot_code/api_based
python main_file.py
```

* Add API keys for the models in the `.env` file.
* `config.py` can be modified to edit the models, prompt and benchmarks as required.

> The same procedure applies for `threeshot_code` experiments.
---

## Results

## Visualisations

