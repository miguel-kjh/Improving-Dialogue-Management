# Analysis of the Impact of Dataset Quality on Task-Oriented Dialogue Management 🗣️

Task-oriented dialogue systems (TODS) have become crucial for users to interact with machines and computers using natural language. One of its key components is the dialogue manager, which guides the conversation towards a good goal for the user by providing the best possible response. Previous works have proposed rule-based systems (RBS), reinforcement learning (RL), and supervised learning (SL) as solutions for the correct dialogue management; in other words, select the best response given input by the user. This work explores the impact of dataset quality on the performance of dialogue managers. We delve into potential errors in popular datasets, such as Multiwoz 2.1 and SGD. For our investigation, we developed a synthetic dialogue generator to regulate the type and magnitude of errors introduced. Our findings suggest that dataset inaccuracies, like mislabeling, might play a significant role in the challenges faced in dialogue management.

## Introduction 📜

We embraced natural language processing methodologies for this project to craft dialogue solutions. We've incorporated data version control and leveraged MongoDB for our primary database needs.

**Note:** Dive deeper into the data structure and the data processing techniques we've employed by visiting this repository:

[**TODS-datasets-processor**](https://github.com/miguel-kjh/TODS-datasets-processor).


---

## Executing the Code with Hydra 🐍

Before running the code, ensure `hydra` and all dependencies from `requirements.txt` are properly installed.

**Steps to execute the code:**

1. Install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main code using Hydra:
   ```bash
   python main.py dataset=multiwoz state=ted model=ted
   ```

The above command executes the code based on the configurations specified in the accompanying `.yaml` files.

## Configuration Insights 📄

Our `.yaml` configuration files detail various model parameters and dataset settings. Here's a brief breakdown:

- **Datasets (`dataset`)**: Specifies the dataset choice. Options span `multi_woz_dataset`, `SGD_dataset`, and `simple_chit_chat_0.1`.

- **Dialogue Manager (`state` and `models`)**: These configurations correspond to the Dialogue State Tracking and Dialogue Policy models, respectively.

Further specifics are embedded within the respective `.yaml` files.

## Experiment Outcomes 📊

### Table 2: Results derived from exhaustive dataset testing.

| Models | **MultiWoz** |  **MultiWoz**     |  **MultiWoz**    | **SGD**  |    **SGD**   |   **SGD**    |
|--------|-------|-------|-------|-------|-------|-------|
|        | F1 %  | Precision % | Recall % | F1 %  | Precision % | Recall % |
| MC     | 39.41 | 54.60       | 34.32    | 73.78 | 77.77       | 71.20    |
| MD     | 35.92 | 51.93       | 30.10    | 78.37 | 90.33       | 72.32    |
| SEQ    | 44.64 | 51.91       | 43.66    | 86.04 | 87.69       | 84.65    |
| RED    | 69.52 | 65.27       | 69.52    | 74.44 | 74.27       | 77.61    |
| TED    | 61.98 | 62.28       | 67.46    | 78.33 | 79.65       | 80.25    |
| PEDP   | 66.95 | 78.11       | 65.02    | 84.74 | 92.07       | 81.30    |

### Table 3: Results derived from synthetic datasets of varying complexities.

| Models | **Simple** |  **Simple**     |  **Simple**     | **Medium** |    **Medium**   |     **Medium**  | **Hard** |    **Hard**    |    **Hard**    |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|        | F1%   | Precision% | Recall% | F1%   | Precision% | Recall% | F1%   | Precision% | Recall% |
| MC     | 85.92 | 91.44      | 84.19   | 86.62 | 92.68      | 84.12   | 85.8  | 91.74      | 83.38   |
| MD     | 81.91 | 89.72      | 80.19   | 80.25 | 90.31      | 77.66   | 80.45 | 90.36      | 77.87   |
| SEQ    | 100   | 100        | 100     | 100   | 100        | 100     | 99.76 | 99.76      | 99.76   |
| RED    | 100   | 100        | 100     | 98.9  | 98.99      | 98.95   | 90.11 | 94.97      | 89.55   |
| TED    | 99.98 | 99.99      | 99.98   | 99.55 | 99.45      | 99.71   | 98.67 | 99.03      | 98.52   |
| PEDP   | 84.85 | 91.27      | 81.5    | 83.08 | 95.95      | 76.57   | 87.55 | 97.45      | 81.56   |

## Tools and Frameworks 🛠️

Our project is powered by several pioneering tools and frameworks to ensure robust and efficient outcomes:

- **MongoDB** 🍃: A versatile document-oriented database. Learn more about [MongoDB](https://www.mongodb.com/).

- **PyTorch Lightning** ⚡: A streamlined PyTorch wrapper for ML enthusiasts. Discover [PyTorch Lightning](https://www.pytorchlightning.ai/).

- **HuggingFace Transformers** 🤗: A state-of-the-art repository offering pre-trained models for varied text tasks. Explore [HuggingFace Transformers](https://huggingface.co/transformers/).

- **Weights & Biases (wandb)** 📊: A platform optimized for deep learning experimentations. Dive into [wandb](https://wandb.ai/site).

Utilizing these tools has been pivotal in ensuring the efficacy, scalability, and reliability of our code.
