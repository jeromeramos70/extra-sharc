# EXtrA-ShaRC

Read our paper at: https://dl.acm.org/doi/10.1145/3627043.3659546

## ShARC Dataset

The raw sharc data can be downloaded using:

```
mkdir data
cd data
wget https://sharc-data.github.io/data/sharc1-official.zip -O sharc_raw.zip
unzip sharc_raw.zip
mv sharc1-official/ sharc_raw
```

To fix some errors in the ShARC dataset, run:

```
python fix_questions.py
```

## EXtrA-ShaRC Dataset

Explanations can be found in `data/explanations.json`

For the scrutability task, the samples can be found in `original_samples.json` and `counterfactual_samples.json` respectively.

## Requirements

Create a new conda environment and run:

```
conda create -n extra-sharc python=3.6
conda install pytorch==1.0.1 cudatoolkit=10.0 -c pytorch
conda install spacy==2.0.16 scikit-learn
python -m spacy download en_core_web_lg && python -m spacy download en_core_web_md
pip install editdistance==0.5.2 transformers==2.8.0
```

## Preprocess data

To preprocess the data, run:

```bash
python preprocess_decision.py
```

## Decision/Explanation

To train the decision and explanation multi-task, run:

```python
python train_sharc.py
--dsave="./out/{}"
--model=base_explain
--data=./data/
--data_type=decision_roberta_base
--prefix=base_explain
--resume=./out/base_explain_new/train_decision/best.pt
--trans_layer=2
```

To evaluate, run:

```python
python train_sharc.py --dsave="./out/{}"
--model=base_explain
--data=./data/
--data_type=decision_roberta_base
--prefix=base_explain
--resume=./out/base_explain_new/train_decision/best.pt
--trans_layer=2
--test
```

## Question Generation

To run the question generation task, run:

```python
python qg.py
```

## Scrutability Task

To run the scrutability tasks, run the evaluation script on a fine-tuned model, but replace the dev set with `data/counterfactual_samples.json` and `data/original_samples.json` during inference.

## Credits

Credits go to https://github.com/Yifan-Gao/Discern and https://github.com/vzhong/e3 for code that we reused and
modified for our new task.

## Reference

To cite our work, please use:

```
@inproceedings{ramos2024extra,
  title={EXtrA-ShaRC: Explainable and Scrutable Reading Comprehension for Conversational Systems},
  author={Ramos, Jerome and Lipani, Aldo},
  booktitle={Proceedings of the 32nd ACM Conference on User Modeling, Adaptation and Personalization},
  pages={47--56},
  year={2024}
}

```
