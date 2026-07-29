"""
Projet CAC 40 - statistiques, beta, LSTM et GRU.

Ce fichier repond aux taches du projet :
  Tache 1  : lecture du fichier cac40.csv avec la fonction read_data (un seul pd.read_csv) ;
  Tache 3  : regroupement des donnees par ticker dans un dictionnaire ;
  Tache 4  : calcul des statistiques (prix moyen, rendements, variance, volatilite) ;
  Tache 5  : estimation du beta et de l'alpha de Jensen par regression OLS ;
  Tache 6  : prediction de la volatilite intraday HL avec des modeles LSTM et GRU ;
  Tache 7  : visualisation des predictions et analyse critique des resultats.

Exemple d'execution :
    python projet_cac40_lstm_gru.py --epochs 8 --lags 20
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


RANDOM_SEED = 42       # Fixed seed for numpy random operations
RISK_FREE_RATE = 0.01  # Annual risk-free rate used in CAPM (Tache 5)

np.random.seed(RANDOM_SEED)

def read_data(path: Path) -> pd.DataFrame:
    """
    Read the CAC 40 CSV file and return a clean typed DataFrame

    :param path: location of cac40.csv
    :return: cleaned market DataFrame sorted by TICKER then DATE
    """
    data = pd.read_csv(path, sep=";", decimal=",", index_col=0)
    data = data.rename(
        columns={
            "jour": "DAY",
            "mois": "MONTH",
            "annee": "YEAR",
        }
    )
    data["DATE"] = pd.to_datetime(
        data[["YEAR", "MONTH", "DAY"]].rename(
            columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}
        )
    )

    numeric_columns = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values(["TICKER", "DATE"]).reset_index(drop=True)
    return data.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])


@dataclass
class SplitData:
    """
    Container for train, validation and test arrays
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_index: pd.DataFrame
    scaler: StandardScaler
    y_scaler: StandardScaler


class MarketAnalyzer:
    """
    Analysis object for CAC 40 prices, returns, statistics and betas
    """

    def __init__(self, data: pd.DataFrame, risk_free_rate: float) -> None:
        """
        Store the dataset and risk-free rate used in beta regressions

        :param data: cleaned CAC 40 DataFrame
        :param risk_free_rate: constant risk-free rate used in CAPM regression
        """
        self.data = data.copy()
        self.risk_free_rate = risk_free_rate
        self.groups_by_ticker: Dict[str, pd.DataFrame] = {}
        self.mean_price_by_ticker: Dict[str, float] = {}
        self.mean_return_by_ticker: Dict[str, float] = {}
        self.variance_by_ticker: Dict[str, float] = {}
        self.volatility_by_ticker: Dict[str, float] = {}
        self.beta_by_ticker: Dict[str, float] = {}
        self.alpha_jensen_by_ticker: Dict[str, float] = {}

    def add_returns(self) -> None:
        """
        Add simple close-to-close returns and High-Low volatility proxy
        """
        self.data["RETURN"] = self.data.groupby("TICKER")["CLOSE"].pct_change()
        self.data["HL"] = self.data["HIGH"] - self.data["LOW"]

    def group_by_ticker(self) -> Dict[str, pd.DataFrame]:
        """
        Group market data by ticker and store each group in a dictionary

        Note: pandas DataFrame.groupby() is used here instead of
        itertools.groupby() because the data is not required to be pre-sorted
        by TICKER and pandas handles the grouping more cleanly

        :return: dictionary mapping tickers to their observations
        """
        self.groups_by_ticker = {
            ticker: group.copy()
            for ticker, group in self.data.groupby("TICKER", sort=True)
        }
        return self.groups_by_ticker

    def compute_statistics(self) -> None:
        """
        Compute mean price, mean return, return variance and volatility
        """
        if not self.groups_by_ticker:
            self.group_by_ticker()

        for ticker, group in self.groups_by_ticker.items():
            returns = group["RETURN"].dropna()
            self.mean_price_by_ticker[ticker] = float(group["CLOSE"].mean())
            self.mean_return_by_ticker[ticker] = float(returns.mean())
            self.variance_by_ticker[ticker] = float(returns.var(ddof=1))
            self.volatility_by_ticker[ticker] = float(returns.std(ddof=1))

    def compute_market_returns(self) -> pd.Series:
        """
        Build the market return rm,t as the average return across assets by date

        :return: daily equally weighted market return series
        """
        return self.data.groupby("DATE")["RETURN"].mean().rename("MARKET_RETURN")

    def compute_betas(self) -> None:
        """
        Estimate alpha and beta for each asset
        """
        market_returns = self.compute_market_returns()
        regression_data = self.data.merge(
            market_returns, left_on="DATE", right_index=True, how="left"
        )

        for ticker, group in regression_data.groupby("TICKER"):
            clean_group = group[["RETURN", "MARKET_RETURN"]].dropna()
            if len(clean_group) < 2:
                self.beta_by_ticker[ticker] = np.nan
                self.alpha_jensen_by_ticker[ticker] = np.nan
                continue

            x_market = clean_group["MARKET_RETURN"].to_numpy()
            y_asset = clean_group["RETURN"].to_numpy()

            # np.polyfit(deg=1) returns [slope, intercept]; slope = beta, intercept = alpha.
            beta, alpha = np.polyfit(x_market, y_asset, deg=1)

            self.beta_by_ticker[ticker] = float(beta)
            self.alpha_jensen_by_ticker[ticker] = float(
                alpha - self.risk_free_rate * (1 - beta)
            )

    def export_summary(self, output_dir: Path) -> pd.DataFrame:
        """
        Save a ticker-level summary table to CSV

        :param output_dir: directory where outputs are saved
        :return: summary DataFrame
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        tickers = sorted(self.groups_by_ticker)
        summary = pd.DataFrame(
            {
                "ticker": tickers,
                "mean_price": [self.mean_price_by_ticker[t] for t in tickers],
                "mean_return": [self.mean_return_by_ticker[t] for t in tickers],
                "variance": [self.variance_by_ticker[t] for t in tickers],
                "volatility": [self.volatility_by_ticker[t] for t in tickers],
                "beta": [self.beta_by_ticker[t] for t in tickers],
                "jensen_alpha": [self.alpha_jensen_by_ticker[t] for t in tickers],
            }
        )
        summary.to_csv(output_dir / "ticker_summary.csv", index=False)
        return summary


class SequenceDatasetBuilder:
    """
    Build lagged sequences of intraday volatility for recurrent models
    """

    def __init__(self, data: pd.DataFrame, lags: int) -> None:
        """
        Save the source data and number of lagged periods

        :param data: market DataFrame containing TICKER, DATE and HL columns
        :param lags: number of lagged High-Low observations in each sequence
        """
        self.data = data
        self.lags = lags

    def _ticker_sequences(
        self, ticker: str, group: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create lagged sequences for a single ticker

        :param ticker: asset ticker
        :param group: observations of this ticker
        :return: X, y and metadata for the target date
        """
        ordered = group.sort_values("DATE").reset_index(drop=True)
        values = ordered["HL"].to_numpy(dtype=float)
        x_values: List[np.ndarray] = []
        y_values: List[float] = []
        index_rows: List[Dict[str, object]] = []

        for position in range(self.lags, len(values)):
            x_values.append(values[position - self.lags : position])
            y_values.append(values[position])
            index_rows.append(
                {
                    "ticker": ticker,
                    "date": ordered.loc[position, "DATE"],
                }
            )

        return (
            np.array(x_values, dtype=float),
            np.array(y_values, dtype=float),
            pd.DataFrame(index_rows),
        )

    def build(self) -> SplitData:
        """
        Build chronological train, validation and test splits for all tickers

        :return: scaled arrays and metadata for the test set
        """
        train_x_list: List[np.ndarray] = []
        train_y_list: List[np.ndarray] = []
        validation_x_list: List[np.ndarray] = []
        validation_y_list: List[np.ndarray] = []
        test_x_list: List[np.ndarray] = []
        test_y_list: List[np.ndarray] = []
        test_index_list: List[pd.DataFrame] = []

        for ticker, group in self.data.groupby("TICKER", sort=True):
            x_ticker, y_ticker, index_ticker = self._ticker_sequences(ticker, group)
            sample_count = len(y_ticker)
            if sample_count < 10:
                continue

            train_end = int(sample_count * 0.70)
            validation_end = int(sample_count * 0.85)

            train_x_list.append(x_ticker[:train_end])
            train_y_list.append(y_ticker[:train_end])
            validation_x_list.append(x_ticker[train_end:validation_end])
            validation_y_list.append(y_ticker[train_end:validation_end])
            test_x_list.append(x_ticker[validation_end:])
            test_y_list.append(y_ticker[validation_end:])
            test_index_list.append(index_ticker.iloc[validation_end:].copy())

        x_train = np.concatenate(train_x_list)
        y_train = np.concatenate(train_y_list)
        x_validation = np.concatenate(validation_x_list)
        y_validation = np.concatenate(validation_y_list)
        x_test = np.concatenate(test_x_list)
        y_test = np.concatenate(test_y_list)
        test_index = pd.concat(test_index_list, ignore_index=True)

        scaler = StandardScaler()
        scaler.fit(x_train.reshape(-1, 1))

        # Dedicated scaler for the targets so that the inverse transform in
        # predict() uses the correct mean and variance for y
        y_scaler = StandardScaler()
        y_scaler.fit(y_train.reshape(-1, 1))

        return SplitData(
            x_train=self._scale_x(x_train, scaler),
            y_train=self._scale_y(y_train, y_scaler),
            x_validation=self._scale_x(x_validation, scaler),
            y_validation=self._scale_y(y_validation, y_scaler),
            x_test=self._scale_x(x_test, scaler),
            y_test=y_test,
            test_index=test_index,
            scaler=scaler,
            y_scaler=y_scaler,
        )

    @staticmethod
    def _scale_x(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        """
        Scale lagged sequences and add the last recurrent dimension

        :param values: sequence array with shape (n_samples, lags)
        :param scaler: fitted scaler
        :return: scaled sequence array with shape (n_samples, lags, 1)
        """
        scaled = scaler.transform(values.reshape(-1, 1)).reshape(values.shape)
        return scaled[..., np.newaxis]

    @staticmethod
    def _scale_y(values: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
        """
        Scale target values with a dedicated scaler fitted on the training targets

        :param values: target array
        :param y_scaler: scaler fitted on y_train
        :return: scaled target array
        """
        return y_scaler.transform(values.reshape(-1, 1)).ravel()


class RecurrentVolatilityModel:
    """
    Wrapper around a Keras recurrent model for High-Low prediction
    """

    def __init__(
        self,
        model_type: str,
        lags: int,
        units: int,
        learning_rate: float,
    ) -> None:
        """
        Build an LSTM or GRU model

        :param model_type: either 'lstm' or 'gru'
        :param lags: sequence length
        :param units: number of recurrent units
        :param learning_rate: Adam optimizer learning rate
        """
        self.model_type = model_type.lower()
        self.lags = lags
        self.units = units
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.training_time_seconds: Optional[float] = None

    def _build_model(self) -> Sequential:
        """
        Build and compile the Keras Sequential architecture for HL prediction

        :return: compiled Keras Sequential model
        """
        if self.model_type == "lstm":
            recurrent_layer = LSTM(self.units)
        elif self.model_type == "gru":
            recurrent_layer = GRU(self.units)
        else:
            raise ValueError("model_type doit etre 'lstm' ou 'gru'.")

        # Architecture: recurrent layer -> light dropout to reduce overfitting
        # -> small dense layer for non-linear projection -> scalar output
        model = Sequential(
            [
                Input(shape=(self.lags, 1)),
                recurrent_layer,
                Dropout(0.15),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",  # MSE loss is consistent with RMSE evaluation metric.
        )
        return model

    def fit(self, split_data: SplitData, epochs: int, batch_size: int) -> None:
        """
        Train the model and store its training duration

        :param split_data: split dataset (only x_train, y_train, x_validation, y_validation are used)
        :param epochs: maximum number of training epochs
        :param batch_size: mini-batch size
        """
        # Stop training if validation loss does not improve for 3 consecutive
        # epochs, and restore the weights from the best epoch.
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
        start_time = time.perf_counter()
        self.model.fit(
            split_data.x_train,
            split_data.y_train,
            validation_data=(split_data.x_validation, split_data.y_validation),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1,
        )
        self.training_time_seconds = time.perf_counter() - start_time

    def predict(self, split_data: SplitData) -> np.ndarray:
        """
        Predict High-Low values on the test set and reverse the scaling

        :param split_data: split dataset (only x_test and y_scaler are used)
        :return: predictions in original High-Low units
        """
        scaled_predictions = self.model.predict(split_data.x_test, verbose=0).ravel()
        return split_data.y_scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)
        ).ravel()


class ResultVisualizer:
    """
    Create plots comparing real and predicted High-Low values
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Store the output directory for generated figures

        :param output_dir: destination directory
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_ticker_predictions(
        self,
        results: pd.DataFrame,
        ticker: str,
    ) -> None:
        """
        Plot observed and predicted High-Low series for one ticker

        :param results: prediction table
        :param ticker: ticker to plot
        """
        ticker_results = results[results["ticker"] == ticker].sort_values("date")
        if ticker_results.empty:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(ticker_results["date"], ticker_results["actual_hl"], label="Reel")
        plt.plot(
            ticker_results["date"],
            ticker_results["lstm_prediction"],
            label="Prediction LSTM",
        )
        plt.plot(
            ticker_results["date"],
            ticker_results["gru_prediction"],
            label="Prediction GRU",
        )
        plt.title(f"Volatilite intraday High-Low - {ticker}")
        plt.xlabel("Date")
        plt.ylabel("High - Low")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"predictions_{ticker}.png", dpi=150)
        plt.close()

    def plot_scatter(self, results: pd.DataFrame) -> None:
        """
        Plot actual values against model predictions for the whole test set

        :param results: prediction table
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(
            results["actual_hl"],
            results["lstm_prediction"],
            s=8,
            alpha=0.35,
            label="LSTM",
        )
        plt.scatter(
            results["actual_hl"],
            results["gru_prediction"],
            s=8,
            alpha=0.35,
            label="GRU",
        )
        min_value = float(results["actual_hl"].min())
        max_value = float(results["actual_hl"].max())
        plt.plot([min_value, max_value], [min_value, max_value], color="black")
        plt.title("Valeurs reelles vs predictions")
        plt.xlabel("High - Low reel")
        plt.ylabel("High - Low predit")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "scatter_predictions.png", dpi=150)
        plt.close()


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE and R2 scores

    :param actual: observed High-Low values
    :param predicted: predicted High-Low values
    :return: dictionary of metrics
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def build_prediction_table(
    split_data: SplitData,
    lstm_predictions: np.ndarray,
    gru_predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Build a DataFrame containing actual and predicted test values

    :param split_data: split arrays and test metadata
    :param lstm_predictions: LSTM predictions in original units
    :param gru_predictions: GRU predictions in original units
    :return: prediction table
    """
    results = split_data.test_index.copy()
    results["actual_hl"] = split_data.y_test
    results["lstm_prediction"] = lstm_predictions
    results["gru_prediction"] = gru_predictions
    # Error = predicted - actual: positive means overestimate, negative means underestimate.
    results["lstm_error"] = results["lstm_prediction"] - results["actual_hl"]
    results["gru_error"] = results["gru_prediction"] - results["actual_hl"]
    return results


def write_conclusion(
    output_dir: Path,
    metrics: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    """
    Save a critical interpretation of the model results

    :param output_dir: directory where outputs are saved
    :param metrics: model metrics table
    :param results: prediction table
    """
    best_row = metrics.sort_values("rmse").iloc[0]
    best_model = best_row["model"]
    best_rmse = best_row["rmse"]
    best_r2 = best_row["r2"]

    worst_lstm = results.assign(abs_error=results["lstm_error"].abs()).nlargest(
        5, "abs_error"
    )
    worst_gru = results.assign(abs_error=results["gru_error"].abs()).nlargest(
        5, "abs_error"
    )

    text = [
        "Conclusion critique",
        "====================",
        "",
        (
            f"Le meilleur modele selon le RMSE est {best_model} "
            f"(RMSE = {best_rmse:.4f}, R2 = {best_r2:.4f}). "
            "Les deux reseaux recurrents utilisent seulement les valeurs passees "
            "de High - Low ; ils capturent donc la dynamique generale, mais les "
            "pics soudains de volatilite restent difficiles a anticiper."
        ),
        "",
        "Periodes avec les plus fortes erreurs LSTM :",
        worst_lstm[["ticker", "date", "actual_hl", "lstm_prediction", "lstm_error"]]
        .to_string(index=False),
        "",
        "Periodes avec les plus fortes erreurs GRU :",
        worst_gru[["ticker", "date", "actual_hl", "gru_prediction", "gru_error"]]
        .to_string(index=False),
        "",
        (
            "Une erreur elevee apparait generalement quand le High - Low observe "
            "augmente brutalement par rapport aux jours precedents. Ce comportement "
            "est coherent avec les marches financiers : une information nouvelle "
            "peut provoquer un saut de volatilite que les seuls lags ne contiennent "
            "pas encore."
        ),
    ]
    (output_dir / "conclusion.txt").write_text("\n".join(text), encoding="utf-8")


def run_project(arguments: argparse.Namespace) -> None:
    """
    Run the full project pipeline.

    :param arguments: command-line arguments
    """
    data_path = Path(arguments.data) if arguments.data else Path(__file__).resolve().parent / "cac40.csv"
    output_dir = Path(arguments.output)

    # Taches 1, 3, 4, 5 : lecture, statistiques et betas
    data = read_data(data_path)
    analyzer = MarketAnalyzer(data=data, risk_free_rate=RISK_FREE_RATE)
    analyzer.add_returns()
    analyzer.group_by_ticker()
    analyzer.compute_statistics()
    analyzer.compute_betas()
    summary = analyzer.export_summary(output_dir)

    print("Apercu des statistiques par ticker :")
    print(summary.head().to_string(index=False))

    # Tache 6 : construction des sequences et entrainement LSTM / GRU
    sequence_builder = SequenceDatasetBuilder(analyzer.data, lags=arguments.lags)
    split_data = sequence_builder.build()

    lstm_model = RecurrentVolatilityModel(
        model_type="lstm",
        lags=arguments.lags,
        units=arguments.units,
        learning_rate=arguments.learning_rate,
    )
    gru_model = RecurrentVolatilityModel(
        model_type="gru",
        lags=arguments.lags,
        units=arguments.units,
        learning_rate=arguments.learning_rate,
    )

    lstm_model.fit(split_data, epochs=arguments.epochs, batch_size=arguments.batch_size)
    gru_model.fit(split_data, epochs=arguments.epochs, batch_size=arguments.batch_size)

    lstm_predictions = lstm_model.predict(split_data)
    gru_predictions = gru_model.predict(split_data)
    results = build_prediction_table(split_data, lstm_predictions, gru_predictions)

    lstm_metrics = evaluate_predictions(split_data.y_test, lstm_predictions)
    gru_metrics = evaluate_predictions(split_data.y_test, gru_predictions)
    metrics = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "rmse": lstm_metrics["rmse"],
                "r2": lstm_metrics["r2"],
                "training_time_seconds": lstm_model.training_time_seconds,
            },
            {
                "model": "GRU",
                "rmse": gru_metrics["rmse"],
                "r2": gru_metrics["r2"],
                "training_time_seconds": gru_model.training_time_seconds,
            },
        ]
    )

    # Tache 7 : sauvegarde des resultats et visualisation
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "predictions_test.csv", index=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    write_conclusion(output_dir, metrics, results)

    visualizer = ResultVisualizer(output_dir)

    # Plot the first N tickers alphabetically as a representative sample.
    selected_tickers = sorted(results["ticker"].unique())[: arguments.plot_tickers]
    for ticker in selected_tickers:
        visualizer.plot_ticker_predictions(results, ticker)
    visualizer.plot_scatter(results)

    print("\nPerformances des modeles :")
    print(metrics.to_string(index=False))
    print(f"\nFichiers sauvegardes dans : {output_dir.resolve()}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the project

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Projet CAC 40 LSTM/GRU")
    parser.add_argument("--data", type=str, default=None, help="Chemin vers cac40.csv")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Dossier de sortie",
    )
    parser.add_argument("--lags", type=int, default=20, help="Nombre de retards p")
    parser.add_argument("--epochs", type=int, default=8, help="Nombre maximal d'epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Taille des batchs")
    parser.add_argument(
        "--units",
        type=int,
        default=32,
        help="Nombre d'unites LSTM/GRU",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taux d'apprentissage Adam",
    )
    parser.add_argument(
        "--plot-tickers",
        type=int,
        default=5,
        help="Nombre de tickers a visualiser",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_project(parse_arguments())
