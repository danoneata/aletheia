from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


def replace_zeros(arr):
    if type(arr) == float:
        if arr == 0:
            arr = np.nan
    else:
        arr[arr == 0] = np.nan

    return arr


def compare_accuracy(df, dataset_name):

    df['score'] = df['score'].apply(
        lambda x: np.fromstring(x.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=' '))
    df['reliab'] = df['reliab'].apply(
        lambda x: np.fromstring(x.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=' '))

    # Flip values
    df['reliab'] = 1 - df['reliab']

    df['reliab'] = df['reliab'].apply(lambda x: np.around(x))
    df['reliab'] = df['reliab'].apply(lambda x: replace_zeros(x))

    df['score_reliab'] = np.array(df['score']) * np.array(df['reliab'])

    df['score_avg'] = df['score'].apply(lambda x: np.nanmean(x))
    df['score_reliab_avg'] = df['score_reliab'].apply(lambda x: np.nanmean(x))

    df['score_bin'] = 0
    df['score_bin'][df['score_avg'] > 0] = 1

    df['score_rel_bin'] = 0
    df['score_rel_bin'][df['score_reliab_avg'] > 0] = 1

    if len(df['label'].unique()) == 2:

        acc_actual = accuracy_score(df[df['label'] == 0]['label'], 1 - df[df['label'] == 0]['score_bin'])
        acc_pred = np.sum(~df[df['label'] == 0]['score_reliab_avg'].isna()) / len(df[df['label'] == 0])
        acc_improved = accuracy_score(df[~df['score_reliab_avg'].isna()][df['label'] == 0]['label'],
                                        1 - df[~df['score_reliab_avg'].isna()][df['label'] == 0]['score_rel_bin'])

        idxs1 = ~df['score_reliab_avg'].isna()
        idxs2 = df['label'] == 0

        num_real = sum(idxs2)
        num_real_kept = sum(idxs1 & idxs2)
        frac_real_kept = 100 * num_real_kept / num_real

        # print(idxs1)
        # print(idxs2)

        # print(df[idxs1][idxs2])
        # print(df[idxs1 & idxs2])
        # import pdb; pdb.set_trace()

        print(
            f'{dataset_name} - REAL - Predicted accuracy: {acc_pred:.2f} - Actual accuracy: {acc_actual:.2f} - Acc. improved: {acc_improved:.2f}')
        print(
            f'Prediction error: {np.abs(acc_pred - acc_actual):.2f} - Improvement: {acc_improved - acc_actual:.2f}')
        print('Fraction kept: {:.2f}'.format(frac_real_kept))
        print()

        acc_actual = accuracy_score(df[df['label'] == 1]['label'], 1 - df[df['label'] == 1]['score_bin'])
        acc_pred = np.sum(~df[df['label'] == 1]['score_reliab_avg'].isna()) / len(df[df['label'] == 1])
        acc_improved = accuracy_score(df[~df['score_reliab_avg'].isna()][df['label'] == 1]['label'],
                                        1 - df[~df['score_reliab_avg'].isna()][df['label'] == 1]['score_rel_bin'])

        idxs1 = ~df['score_reliab_avg'].isna()
        idxs2 = df['label'] == 1

        num_fake = sum(idxs2)
        num_fake_kept = sum(idxs1 & idxs2)
        frac_fake_kept = 100 * num_fake_kept / num_fake

        print(
            f'{dataset_name} - FAKE - Predicted accuracy: {acc_pred:.2f} - Actual accuracy: {acc_actual:.2f} - Acc. improved: {acc_improved:.2f}')
        print(
            f'Prediction error: {np.abs(acc_pred - acc_actual):.2f} - Improvement: {acc_improved - acc_actual:.2f}')
        print('Fraction kept: {:.2f}'.format(frac_fake_kept))
        print()

        acc_actual = accuracy_score(df['label'], 1 - df['score_bin'])
        acc_pred = np.sum(~df['score_reliab_avg'].isna()) / len(df)
        acc_improved = accuracy_score(df[~df['score_reliab_avg'].isna()]['label'],
                                        1 - df[~df['score_reliab_avg'].isna()]['score_rel_bin'])

        idxs1 = ~df['score_reliab_avg'].isna()

        num = len(df)
        num_kept = sum(idxs1)
        frac_kept = 100 * num_kept / num

        print(
            f'{dataset_name} - ALL - Predicted accuracy: {acc_pred:.2f} - Actual accuracy: {acc_actual:.2f} - Acc. improved: {acc_improved:.2f}')
        print(
            f'Prediction error: {np.abs(acc_pred - acc_actual):.2f} - Improvement: {acc_improved - acc_actual:.2f}')
        print('Fraction kept: {:.2f}'.format(frac_kept))
        print()



    else:

        acc_actual = accuracy_score(df['label'], 1 - df['score_bin'])
        acc_pred = np.sum(~df['score_reliab_avg'].isna()) / len(df)
        acc_improved = accuracy_score(df[~df['score_reliab_avg'].isna()]['label'],
                                        1 - df[~df['score_reliab_avg'].isna()]['score_rel_bin'])

        print(
            f'{dataset_name} - Predicted accuracy: {acc_pred:.2f} - Actual accuracy: {acc_actual:.2f} - Acc. improved: {acc_improved:.2f}')
        print(
            f'Prediction error: {np.abs(acc_pred - acc_actual):.2f} - Improvement: {acc_improved - acc_actual:.2f}')
        print()


import sys
dataset_name = sys.argv[1]  # "in-the-wild"
path = f"output/salvi-{dataset_name}.csv"
df = pd.read_csv(path, index_col=0)
compare_accuracy(df, dataset_name)