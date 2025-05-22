# Lesson 05: Semantic Vector Extraction and Word Segmentation

This repository contains the materials and code for "Lesson 05", focusing on natural language processing tasks, specifically semantic vector extraction and word segmentation. It includes scripts for data processing, model utilities, and a Jupyter notebook for experimentation and analysis.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)

## Project Structure

The project is organized as follows:
```txt
├── data/                    # Directory for raw and processed data files.
├── log/                     # Directory for log files generated during script execution.
├── model/                   # Directory to store trained models or model-related artifacts.
├── src/                     # Source code directory
│   ├── utils/               # Utility functions and helper scripts.
│   ├── __init__.py
│   ├── extract_semanticVector.py    # Script for extracting semantic vectors from text.
│   └── segment_word.py         # Script for performing word segmentation.
├── lab-5.pdf                # Lab instructions for lesson 05.
├── notebook.ipynb           # Jupyter notebook for demonstrating code (lab report).
├── requirements.txt         # Python dependencies required for the project.
├── .gitattributes           # Git attributes configuration.
├── .gitignore               # Files and directories to be ignored by Git.
└── README.md                # This README file.
```
## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

* Python 3.x (`python 3.10.x` is recommended)
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/nguyenlong205/uit-ds102-practicalHomework
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python310 -m venv venv
    ```

3.  **Activate the virtual environment:**

    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project provides scripts for specific NLP tasks (it **MUST** be compliled respectively).
* To perform word segmentation, run `word_segmentation.py`.
* To extract semantic vectors, run `extract_semanticVector.py`.

