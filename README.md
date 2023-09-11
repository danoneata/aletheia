# Aletheia ICASSP24

Code for our ICASSP'24 submission.

This project addresses the task deepfake audio detection.
The name of the project, "Alteheia", comes from Greek and means "unconcealedness", "disclosure", "revealing"; see [the corresponding Wikipedia entry](https://en.wikipedia.org/wiki/Aletheia) for more information.

## Set up

```bash
pip install -e .
```

## Extract features

Extract features for the test sets; for example:

```bash
for d in asvspoof19 in-the-wild; do
    python aletheia/scripts/extract_features.py -d $d -s test -f wav2vec2-xls-r-2b
done
```

Extract features for the train sets; for example:

```bash
for split in train valid; do
    for num in 1000 2000 4000 8000; do
        for seed in 0 1 2; do
            python aletheia/scripts/extract_features.py -d asvspoof19 -s ${split} -f wav2vec2-xls-r-2b --subset ${num}-${seed}
        done
    done
done
```

## Train and evaluate model

Table 2 in paper:
```bash
python aletheia/scripts/evaluate_feature_type.py
```

Figure 1 in  paper:
```bash
streamlit run aletheia/scripts/plot_num_training_samples.py
```

Figure 2 (uncertainty estimation and reliability) in paper:
```bash
streamlit run aletheia/scripts/evaluate_reliability_ours_vs_salvi.py
```