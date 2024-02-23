---
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- generated_from_trainer
model-index:
- name: url_classifier
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# url_classifier

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.8171

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- training_steps: 500

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 3.2484        | 0.05  | 25   | 3.0517          |
| 2.929         | 0.11  | 50   | 2.7754          |
| 2.6472        | 0.16  | 75   | 2.6031          |
| 2.4557        | 0.21  | 100  | 2.4572          |
| 2.3967        | 0.26  | 125  | 2.3530          |
| 2.4517        | 0.32  | 150  | 2.2555          |
| 2.2021        | 0.37  | 175  | 2.1556          |
| 2.1284        | 0.42  | 200  | 2.0751          |
| 1.8457        | 0.48  | 225  | 2.0418          |
| 1.9308        | 0.53  | 250  | 2.0402          |
| 2.1185        | 0.58  | 275  | 2.0035          |
| 1.9027        | 0.63  | 300  | 1.9822          |
| 1.9469        | 0.69  | 325  | 1.9410          |
| 2.0847        | 0.74  | 350  | 1.9050          |
| 1.9513        | 0.79  | 375  | 1.8948          |
| 1.8126        | 0.85  | 400  | 1.8771          |
| 2.1342        | 0.9   | 425  | 1.8413          |
| 2.0807        | 0.95  | 450  | 1.8250          |
| 1.918         | 1.0   | 475  | 1.8200          |
| 1.5948        | 1.06  | 500  | 1.8171          |


### Framework versions

- Transformers 4.37.1
- Pytorch 2.1.2
- Datasets 2.16.1
- Tokenizers 0.15.1
