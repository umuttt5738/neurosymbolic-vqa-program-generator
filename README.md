# Neurosymbolic VQA Program Generator

This project, developed for the System 2 AI course, implements a comprehensive Neurosymbolic framework for Visual Question Answering (VQA). It focuses on translating natural language questions from the CLEVR dataset into executable symbolic programs. We explore and compare three distinct learning paradigms to train this neural-to-symbolic translator: Supervised Learning (with LSTM and Transformer models), Reinforcement Learning (using REINFORCE), and In-Context Learning (with a pre-trained LLM).

## Features

* **Multi-Paradigm Training**: Implements Supervised, REINFORCE, and In-Context Learning (ICL) strategies.
* **Dual Architectures**: Provides both LSTM-based (with attention) and Transformer-based Seq2Seq models for the core generator.
* **Symbolic Executor**: Includes a `ClevrExecutor` that runs the generated programs on scene data to produce concrete answers.
* **Modular & Extensible**: Fully modular code for data preprocessing, model definitions, training loops, and evaluation.
* **Detailed Guides**: A suite of Jupyter Notebooks provides a clean, step-by-step guide for running all experiments.

## Core Concepts & Techniques

* **Neurosymbolic AI**: Bridges the gap between connectionist (neural) models for perception and classic symbolic (logical) models for reasoning.
* **Sequence-to-Sequence (Seq2Seq)**: The core framework used to translate question-sequences into program-sequences.
* **Policy Gradients (REINFORCE)**: An RL algorithm used to fine-tune the model based on the *semantic correctness* (right answer) rather than just *syntactic correctness* (right program).
* **In-Context Learning (ICL)**: A "zero-training" approach that leverages the pattern-recognition abilities of Large Language Models (LLMs) by providing examples in a prompt.
* **Visual Question Answering (VQA)**: The target task, requiring the model to reason about visual scenes to answer natural language questions.

---

## How It Works

This project's goal is to solve a VQA task by generating a symbolic program that represents the reasoning steps needed to answer a question.

### 1. The Neurosymbolic Approach: An Overview

The core idea is to decouple the problem:

1.  **Neural Perception/Translation**: A Seq2Seq model (the "neural" part) reads the ambiguous natural language question and translates it into a structured, logical program.
    * **Question**: "Is there a small rubber cube behind the green cylinder?"
    * **Program**: `scene` $\rightarrow$ `filter_shape[cylinder]` $\rightarrow$ `filter_color[green]` $\rightarrow$ `unique` $\rightarrow$ `relate[behind]` $\rightarrow$ `filter_shape[cube]` $\rightarrow$ `filter_material[rubber]` $\rightarrow$ `filter_size[small]` $\rightarrow$ `exist`
2.  **Symbolic Reasoning/Execution**: A symbolic executor (the "symbolic" part) takes this unambiguous program and executes it step-by-step against a structured representation of the scene to get the final answer.
    * `scene` $\rightarrow$ (gets all objects)
    * `filter_shape[cylinder]` $\rightarrow$ (keeps only cylinders)
    * ...
    * `exist` $\rightarrow$ (checks if the final set of objects is empty) $\rightarrow$ **"yes"**

This project focuses entirely on **Step 1**: building and training the best possible program generator.

### 2. Data & Program Representation

* **Input Data**: We use the **CLEVR dataset**, which consists of images (scenes) containing simple shapes (cubes, spheres, cylinders) of different sizes, colors, and materials.
* **Questions**: Natural language questions about the scenes (e.g., "How many...", "Are there...").
* **Programs**: The ground-truth data provides a "functional program" for each question. We preprocess these programs into a single string (in prefix notation by default) which becomes the target sequence for our models.
    * e.g., `<START> exist filter_size[small] filter_material[rubber] ... <END>`

### 3. Implemented Learning Strategies

We implement and compare three different ways to train the program generator.

#### A. Supervised Learning (Seq2Seq)

This is the baseline "behavioral cloning" approach. The model is trained to minimize the cross-entropy loss between its prediction and the ground-truth program, one token at a time.

* **Objective**: Maximize the probability of the ground-truth program $Y$ given the question $X$.
* **Loss Function**: Standard Cross-Entropy Loss (Teacher Forcing).

  $$L_{\text{SL}} = - \sum_{t=1}^{T} \log p(y_t | y_{<t}, \mathbf{X}; \theta)$$
  
* **Problem**: This leads to **exposure bias**. The model is only trained on perfect, ground-truth prefixes. At inference, if it makes a single mistake, it may enter a state it has never seen, causing errors to cascade.

#### B. Reinforcement Learning (REINFORCE)

This approach fine-tunes the supervised model to solve the exposure bias problem. Instead of forcing the model to match a specific program, we reward it for producing *any* program that gets the right answer.

* **Policy ($\pi_{\theta}$)**: Our Seq2Seq model.
* **Action ($a$)**: A full program *sampled* from the model's output distribution.
* **Reward ($R$)**: We run the sampled program $a$ through the `ClevrExecutor`.
    * $R = 1.0$ if the program's answer matches the ground-truth answer.
    * $R = 0.0$ otherwise.
* **Baseline ($b$)**: To stabilize training, we use a moving average of past rewards. The **Advantage** is $A = (R - b)$.
* **Objective**: We use the REINFORCE algorithm to update the model's weights ($\theta$) to maximize the expected reward. The loss is the negative policy gradient objective:

  $$L_{\text{RL}} = - \mathbb{E}\_{a \sim \pi_{\theta}} [ (R - b) \sum_{t=1}^{T} \log \pi_{\theta}(a_t | a_{<t}, \mathbf{X}) ]$$

  This "pushes up" the probability of programs that lead to a positive advantage and "pushes down" the probability of those that lead to a negative one.

#### C. In-Context Learning (ICL)

This modern approach uses a large, pre-trained LLM and requires **no training or fine-tuning**. We leverage the LLM's powerful pattern-matching abilities by "showing" it examples of the task in its prompt.

* **Prompt**: We construct a prompt containing a system message and $k$ "shots" (examples).
* **Zero-Shot ($k=0$)**: The prompt only contains the instructions.
* **Few-Shot ($k>0$)**: The prompt contains instructions *and* $k$ examples of `(Question, Program)` pairs.
* **Evaluation**: We test the LLM's ability to generate a correct program for a new, unseen question. We evaluate its performance as $k$ increases to see how quickly it "learns" the task.

---

## Project Structure

```
neurosymbolic-vqa-program-generator/
├── .gitignore                               # Ignores data, logs, models, and Python caches
├── LICENSE                                  # MIT License
├── README.md                                # This file
├── requirements.txt                         # Python dependencies
├── data/
│   └── .gitkeep                             # Placeholder for data/ (CLEVR\_Dataset/ goes here)
├── logs/
│   └── .gitkeep                             # Placeholder for logs/ (e.g., train.log)
├── models/
│   └── .gitkeep                             # Placeholder for models/ (e.g., lstm.pth)
├── notebooks/
│   ├── 0_Data_Exploration_and_Setup.ipynb   # Guide: Download & preprocess data
│   ├── 1_Supervised_Training.ipynb          # Guide: Run supervised experiments
│   ├── 2_Reinforce_Finetuning.ipynb         # Guide: Run RL experiments
│   └── 3_ICL_Evaluation.ipynb               # Guide: Run LLM ICL experiments
├── scripts/
│   ├── preprocess_data.py                   # Runnable script to build vocab & H5 files
│   ├── train.py                             # Main entry point for training (Supervised & RL)
│   └── evaluate.py                          # Runnable script to evaluate a trained model
└── src/
├── __init__.py
├── config.py                                # Stores all paths and hyperparameters
├── data_loader.py                           # Contains ClevrQuestionDataset and DataLoader
├── executor.py                              # Contains the ClevrExecutor for running programs
├── vocabulary.py                            # Helper functions for building/loading vocabs
├── models/
│   ├── __init__.py
│   ├── base_rnn.py                          # Base class for RNNs
│   ├── lstm_seq2seq.py                      # LSTM Encoder, Decoder, and Attention
│   └── transformer_seq2seq.py               # Transformer model implementation
├── training/
│   ├── __init__.py
│   ├── train_reinforce.py                   # Trainer class for the REINFORCE loop
│   └── train_supervised.py                  # Trainer class for the Supervised loop
├── evaluation/
│   ├── __init__.py
│   ├── eval_icl.py                          # Logic for running ICL evaluation
│   └── eval_model.py                        # Logic for evaluating our trained models
└── utils/
    ├── __init__.py
    ├── logger.py                            # Sets up file and console logging
    ├── program_utils.py                     # Helpers for program list/tree/string conversions
    └── scene_utils.py                       # Helper for loading scene JSONs
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/neurosymbolic-vqa-program-generator.git
    cd neurosymbolic-vqa-program-generator
    ```

2.  **Setup Environment and Data:**
    ```bash
    # Install dependencies
    pip install -r requirements.txt
    
    # Download the CLEVR dataset (see Notebook 0 for link)
    # Unzip and place it in the `data/` folder.
    # The path should be: data/CLEVR_Dataset/
    ```

3.  **Run Preprocessing:**
    First, preprocess the `train` data to create the vocabulary. Then, use that vocabulary to preprocess the `val` and `test` data.

    ```bash
    # Process TRAIN (creates vocab)
    python scripts/preprocess_data.py \
        --input_json data/CLEVR_Dataset/Questions/CLEVR_train_questions.json \
        --output_h5 data/dataH5Files/clevr_train_questions.h5 \
        --output_vocab_json data/dataH5Files/clevr_vocab.json

    # Process VAL (uses existing vocab)
    python scripts/preprocess_data.py \
        --input_json data/CLEVR_Dataset/Questions/CLEVR_val_questions.json \
        --input_vocab_json data/dataH5Files/clevr_vocab.json \
        --output_h5 data/dataH5Files/clevr_val_questions.h5 \
        --allow_unk 1
    ```

4.  **Run Experiments (via Notebooks):**
    The easiest way to run the project is to follow the Jupyter Notebooks in the `notebooks/` directory.

    ```bash
    jupyter lab notebooks/
    ```
    * **`0_Data_Exploration_and_Setup.ipynb`**: Confirms data is set up correctly.
    * **`1_Supervised_Training.ipynb`**: Runs supervised training for both models.
    * **`2_Reinforce_Finetuning.ipynb`**: Runs REINFORCE fine-tuning.
    * **`3_ICL_Evaluation.ipynb`**: Runs the final LLM-based evaluation.

5.  **Run Experiments (via Terminal):**
    You can also run the scripts directly from the command line.

    ```bash
    # Example: Supervised Training (LSTM)
    python scripts/train.py \
        --model_type lstm \
        --train_mode supervised \
        --model_save_path models/supervised_lstm.pth \
        --num_iters 100000

    # Example: REINFORCE Fine-Tuning (LSTM)
    python scripts/train.py \
        --model_type lstm \
        --train_mode reinforce \
        --load_model models/supervised_lstm.pth \
        --model_save_path models/reinforce_lstm.pth \
        --num_iters 50000 \
        --learning_rate 1e-5

    # Example: Evaluate a model
    python scripts/evaluate.py \
        --model_type lstm \
        --model_path models/reinforce_lstm.pth \
        --data_h5_path data/dataH5Files/clevr_val_questions.h5 \
        --split val
    ```

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
