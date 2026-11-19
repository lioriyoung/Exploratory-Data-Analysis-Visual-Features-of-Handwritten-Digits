"""Tests for EDA data preparation and summaries."""
import pandas as pd
from eda import load_data, summarize


def test_data_shape():
    df = load_data()
    assert len(df) == 1797
    assert df['digit'].nunique() == 10
    assert df.isna().sum().sum() == 0
    assert df['mean_intensity'].between(0, 16).all()
    assert df['active_pixels'].between(0, 64).all()


def test_summary():
    df = pd.DataFrame({'mean_intensity': [1, 2, 3],
                       'active_pixels': [2, 4, 6],
                       'upper_intensity': [0, 2, 4],
                       'lower_intensity': [2, 2, 2]})
    result = summarize(df)
    assert result.loc['mean', 'mean_intensity'] == 2
    assert result.loc['50%', 'active_pixels'] == 4
    assert result.loc['max', 'lower_intensity'] == 2


if __name__ == '__main__':
    test_data_shape()
    test_summary()
    print('All tests passed.')
