# Introduction

This repository contains the source code for the Joint Emotion and Cognitive Topic Model (JECTM) and BERT-based classification models as described in the associated manuscript.

## Joint Emotion and Cognitive Topic Model (JECTM)

The JECTM model is implemented in Java.

### Key Files:

- `JECTM/Test/JECTM.java`: Main entry point for running the JECTM model
- `JECTM/Model/JECTM_Model.java`: Core implementation of the JECTM model
- `JECTM/eval/CoherenceEval.java`: Code for evaluating topic coherence

### Usage:

To run the JECTM model, execute the `JECTM.java` file. Make sure to set up the required data and parameters as specified in the code comments.

## BERT-based Classification

This section implements BERT and other transformer-based models for emotion and cognitive engagement classification.

### Key Files:

- `training.py`: Main script for training the classification models
- `train_eval.py`: Contains training and evaluation functions
- `predict_pretrained.py`: Script for inference using trained models

### Training and Evaluation:

To train a model:

```
python training.py --model <Bert>
```


### Inference:

To run inference on new data:

```
python predict_pretrained.py --model <Bert>
```


This script will load a trained model and make predictions on the data specified in the script.

## Data

The models expect data in specific formats:
- JECTM: Excel files containing learner achievement and discussion data
- BERT: CSV or Excel files with text content and labels

Refer to the individual script comments for more details on data formatting and paths.

## Requirements

- Java (for JECTM)
- Python 3.8
- PyTorch
- Transformers library
- Pandas, NumPy, and other common Python data science libraries

## Citation

If you use this code in your research, please cite our paper:

[Include citation information here]

## License

[Specify the license under which this code is released]