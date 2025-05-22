# Lesson 05: Word Segmentation, Semantic Vector Extraction and Gaussian Mixture Model

> This repository contains the assignment for lesson 05 (practical lesson - *DS102 - Statistical Machine Learning*), focusing on the application of the **Gaussian Mixture Model (GMM)** for classification tasks in Vietnamese NLP. The project includes preprocessing steps such as *word segmentation* using `py_vncorenlp` and semantic vector extraction using the `PhoBERT` model. These vectors serve as input features for GMM-based prediction, with supporting scripts and a Jupyter notebook for implementation and analysis.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Assignment description](#assignment-description)

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

## Assignment description

This project is the assignments to my lecturer at my practical lesson. It includes scripts for specific assigned tasks (it **MUST** be compliled respectively).
* **Task 01**: To perform word segmentation, run `word_segmentation.py`. It then automatically generate `wordSegmentedDatasets` folder, encompassing neccessary datasets for task 02.
* **Task 02**: To extract semantic vectors, run `extract_semanticVector.py`. It then automatically generate `semanticVectors` folder, encompassing neccessary datasets for task 03.

