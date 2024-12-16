import os

from tsfresh import extract_features
import tsfresh
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.feature_selection import SelectKBest
from lightgbm import LGBMClassifier

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

useful_features_dict = {
    "blue": {
        "index_mass_quantile": [{"q": 0.1}, {"q": 0.2}, {"q": 0.3}, {"q": 0.4}],
        "cwt_coefficients": [
            {"coeff": 1, "w": 10, "widths": (2, 5, 10, 20)},
            {"coeff": 2, "w": 10, "widths": (2, 5, 10, 20)},
        ],
        "linear_trend": [{"attr": "rvalue"}],
    },
    "green": {
        "index_mass_quantile": [{"q": 0.3}, {"q": 0.4}],
        "linear_trend": [{"attr": "rvalue"}],
    },
    "red": {
        "index_mass_quantile": [{"q": 0.1}, {"q": 0.2}, {"q": 0.3}, {"q": 0.4}],
        "cwt_coefficients": [
            {"coeff": 0, "w": 10, "widths": (2, 5, 10, 20)},
            {"coeff": 1, "w": 10, "widths": (2, 5, 10, 20)},
            {"coeff": 2, "w": 10, "widths": (2, 5, 10, 20)},
        ],
        "fft_coefficient": [{"attr": "real", "coeff": 1}],
    },
    "red_edge_2": {
        "mean_abs_change": None,
        "absolute_sum_of_changes": None,
        "change_quantiles": [
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0},
            {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.2},
        ],
        "fft_coefficient": [{"attr": "real", "coeff": 1}],
        "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 3}],
    },
    "red_edge_3": {
        "mean_abs_change": None,
        "standard_deviation": None,
        "variance": None,
        "absolute_sum_of_changes": None,
        "cid_ce": [{"normalize": False}],
        "index_mass_quantile": [{"q": 0.2}],
        "change_quantiles": [
            {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.0},
            {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.0},
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0},
            {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.2},
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.2},
            {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.4},
        ],
        "fft_coefficient": [{"attr": "real", "coeff": 1}, {"attr": "abs", "coeff": 1}],
        "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 3}],
    },
    "nir": {
        "mean_abs_change": None,
        "standard_deviation": None,
        "variance": None,
        "absolute_sum_of_changes": None,
        "cid_ce": [{"normalize": False}],
        "index_mass_quantile": [{"q": 0.2}],
        "change_quantiles": [
            {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.0},
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0},
            {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.2},
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.2},
        ],
        "fft_coefficient": [{"attr": "real", "coeff": 1}],
        "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 3}],
    },
    "red_edge_4": {
        "mean_abs_change": None,
        "standard_deviation": None,
        "variance": None,
        "absolute_sum_of_changes": None,
        "index_mass_quantile": [{"q": 0.1}, {"q": 0.2}, {"q": 0.3}],
        "change_quantiles": [
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0},
            {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.2},
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.2},
        ],
        "fft_coefficient": [{"attr": "real", "coeff": 1}, {"attr": "abs", "coeff": 1}],
        "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 3}],
    },
    "swir_1": {
        "index_mass_quantile": [{"q": 0.2}, {"q": 0.3}, {"q": 0.4}],
        "change_quantiles": [
            {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.2},
            {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.2},
        ],
        "fft_coefficient": [{"attr": "imag", "coeff": 1}],
    },
    "swir_2": {
        "index_mass_quantile": [{"q": 0.1}, {"q": 0.2}, {"q": 0.3}, {"q": 0.4}],
        "cwt_coefficients": [
            {"coeff": 0, "w": 10, "widths": (2, 5, 10, 20)},
            {"coeff": 1, "w": 10, "widths": (2, 5, 10, 20)},
            {"coeff": 2, "w": 10, "widths": (2, 5, 10, 20)},
        ],
        "change_quantiles": [
            {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.2},
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.2},
            {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.4},
        ],
        "fft_coefficient": [{"attr": "imag", "coeff": 1}],
        "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 1}],
    },
}

df = pd.read_parquet("data/california_sits_bert_original.parquet")

df.drop(columns=["label", "use_bert"], inplace=True)
features = extract_features(
    df, column_id="id", column_sort="time", kind_to_fc_parameters=useful_features_dict
)

features.to_parquet("data/california_sits_bert_features.parquet", compression="brotli")
