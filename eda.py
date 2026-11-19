"""CSE 163 exploratory data analysis of real handwritten digit images."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits


OUTPUT = Path(__file__).resolve().parent


def load_data():
    """Return one row per digit image, with pixel and derived features."""
    digits = load_digits()
    pixels = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
    pixels['digit'] = digits.target
    pixels['mean_intensity'] = pixels[[f'pixel_{i}' for i in range(64)]].mean(axis=1)
    pixels['active_pixels'] = (pixels[[f'pixel_{i}' for i in range(64)]] > 0).sum(axis=1)
    pixels['upper_intensity'] = pixels[[f'pixel_{i}' for i in range(32)]].mean(axis=1)
    pixels['lower_intensity'] = pixels[[f'pixel_{i}' for i in range(32, 64)]].mean(axis=1)
    return pixels


def summarize(df):
    """Return seven-number summaries for the quantitative variables."""
    cols = ['mean_intensity', 'active_pixels', 'upper_intensity',
            'lower_intensity']
    return df[cols].describe().loc[
        ['mean', 'std', 'min', '25%', '50%', '75%', 'max']]


def make_plots(df):
    """Save plots showing category distribution and image features."""
    counts = df['digit'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values)
    ax.set(title='Number of Images per Digit', xlabel='Digit label',
           ylabel='Number of images')
    ax.set_xticks(range(10))
    fig.tight_layout()
    fig.savefig(OUTPUT / 'figure1_counts.png', dpi=180)
    plt.close(fig)

    grouped = [df.loc[df['digit'] == d, 'mean_intensity'] for d in range(10)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(grouped, tick_labels=[str(d) for d in range(10)])
    ax.set(title='Average Pixel Intensity by Digit', xlabel='Digit label',
           ylabel='Mean pixel intensity (0–16)')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'figure2_intensity.png', dpi=180)
    plt.close(fig)

    means = df.groupby('digit')[['upper_intensity', 'lower_intensity']].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot(kind='bar', ax=ax)
    ax.set(title='Upper vs Lower Image Intensity', xlabel='Digit label',
           ylabel='Average pixel intensity (0–16)')
    ax.legend(['Upper half', 'Lower half'])
    fig.tight_layout()
    fig.savefig(OUTPUT / 'figure3_halves.png', dpi=180)
    plt.close(fig)


def main():
    df = load_data()
    print('Rows, columns:', df.shape)
    print('Missing values:', int(df.isna().sum().sum()))
    print('Digit counts:\n', df['digit'].value_counts().sort_index())
    print('Seven-number summary:\n', summarize(df))
    print('Per-digit means:\n', df.groupby('digit')[
        ['mean_intensity', 'active_pixels', 'upper_intensity',
         'lower_intensity']].mean())
    make_plots(df)


if __name__ == '__main__':
    main()
