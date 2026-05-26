# LLM Workshop Exercises

This repository contains hands-on exercises for learning about Large Language Models (LLMs). The material is divided into 5 interactive Jupyter notebooks that progressively explore core concepts.

## Exercises Overview

| Notebook | Topic | What You'll Learn |
|----------|-------|-------------------|
| 1. Tokenization | Subword Tokenization & BPE | How text is converted to tokens using Byte Pair Encoding, explore GPT-2 vocabulary, understand token IDs and special characters |
| 2. Embeddings | Word Embeddings | Compute semantic similarities between words using GloVe vectors, perform vector arithmetic, understand how embeddings capture meaning |
| 3. Decoding | Sampling Strategies | Explore temperature, top-K, and top-P sampling; visualize token probability distributions and perplexity |
| 4. Prompting | Prompt Engineering | Learn chain-of-thought reasoning, self-guidance techniques, in-context learning with few-shot examples |
| 5. Classification | LLM for Classification | Build a classification system using LLMs, extract token probabilities for choice selection, evaluate performance with confusion matrices |

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llm-workshop
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the `exercises` directory with your OpenAI API key:
   ```env
   OPENAI_BASE_URL=...
   OPENAI_API_KEY=...
   ```

## Getting Started

1. **Launch Jupyter Lab:**
   ```bash
   cd exercises
   jupyter lab
   ```

2. **Open a notebook:**
    - Navigate to the `exercises` directory in the file browser
    - Double-click on any `.ipynb` file to open it
    - Work through the exercises in order (1 through 5) for a progressive learning experience

Each notebook contains interactive code cells, exercises with `<TODO>` placeholders, and visualizations to help you understand LLM concepts hands-on.