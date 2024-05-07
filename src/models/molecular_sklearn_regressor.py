from pathlib import Path
from typing import Optional, Sequence

import joblib  # type: ignore [import-untyped]
import numpy as np
from molfeat.trans import MoleculeTransformer  # type: ignore [import-untyped]
from sklearn.base import BaseEstimator  # type: ignore [import-untyped]
from sklearn.base import RegressorMixin, TransformerMixin


class MolecularSKLearnRegressor:
    """A class representing a molecular SKLearn regressor.

    Args:
        sklearn_regressor (BaseEstimator): The predictive regression model being wrapped.
        molecule_transformer (MoleculeTransformer): Transformer that turns smiles into fingerprints or descriptors.
        scaler (Optional[BaseEstimator]): An sklearn scaler object that is applied to the fingerprints/descriptors. (optional).

    Raises:
        TypeError: If the `sklearn_regressor` is not an instance of `BaseEstimator`.
        TypeError: If the `sklearn_regressor` is not an instance of `RegressorMixin`.
        TypeError: If the `molecule_transformer` is not an instance of `MoleculeTransformer`.
        TypeError: If the `scaler` is provided but not an instance of `BaseEstimator`.
        TypeError: If the `scaler` is provided but not an instance of `TransformerMixin`.

    Attributes:
        model (BaseEstimator): The predictive regression model being wrapped.
        featurizer (MoleculeTransformer): Transformer that turns smiles into fingerprints or descriptors.
        scaler (Optional[BaseEstimator]):  An sklearn scaler object that is applied to the fingerprints/descriptors. (optional).

    """  # noqa: E501

    def __init__(
        self,
        sklearn_regressor: BaseEstimator,
        molecule_transformer: MoleculeTransformer,
        scaler: Optional[BaseEstimator],
    ):
        if not isinstance(sklearn_regressor, BaseEstimator):
            raise TypeError(
                f"{type(sklearn_regressor)=} but must inherit sklearn.base.BaseEstimator"
            )
        if not isinstance(sklearn_regressor, RegressorMixin):
            raise TypeError(
                f"{type(sklearn_regressor)=} but must inherit sklearn.base.RegressorMixin"
            )
        if not isinstance(molecule_transformer, MoleculeTransformer):
            raise TypeError(
                f"{type(molecule_transformer)=} but must be MoleculeTransformer"
            )
        if scaler:
            if not isinstance(scaler, BaseEstimator):
                raise TypeError(
                    f"{type(scaler)=} but must inherit sklearn.base.BaseEstimator"
                )
            if not isinstance(scaler, TransformerMixin):
                raise TypeError(
                    f"{type(scaler)=} but must inherit sklearn.base.TransformerMixin"
                )

        self.model = sklearn_regressor
        self.featurizer = molecule_transformer
        self.scaler = scaler

    @classmethod
    def from_pretrained(
        cls,
        sklearn_model_path: Path,
        featurizer_yaml_path: Path,
        scaler_path: Optional[Path],
    ):
        """
        Creates a new instance of the MolecularSklearnRegressor class from a
        pretrained model that has been saved to disc.

        Args:
            cls (type): The class object.
            sklearn_model_path (Path): The path to the pretrained sklearn model file.
            featurizer_yaml_path (Path): The path to the featurizer YAML file.
            scaler_path (Optional[Path]): The path to the scaler file (optional).

        Returns:
            MolecularSklearnRegressor: A new instance of the MolecularSklearnRegressor class.

        """  # noqa: E501
        sklearn_regressor = joblib.load(sklearn_model_path)
        molecular_transformer = MoleculeTransformer.from_state_yaml_file(
            str(featurizer_yaml_path)
        )
        if scaler_path:
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
        return cls(
            sklearn_regressor=sklearn_regressor,
            molecule_transformer=molecular_transformer,
            scaler=scaler,
        )

    def featurize_data(self, X_raw: Sequence[str]) -> np.ndarray:
        """
        Featurize smiles into molecular fingerprints.
        Args:
            X_raw (Sequence[str]): A list of SMILES strings.

        Returns:
            X_features (np.ndarray): A numpy array of molecular fingerprints/descriptors.
        """
        X_features = self.featurizer(X_raw)
        return X_features

    def scale_data(self, X_features: np.ndarray) -> np.ndarray:
        """
        Scale the molecular fingerprints/descriptors, if a scaler has been provided.

        Args:
            X_features (np.ndarray): Input features to be scaled.

        Returns:
            X_scaled (np.ndarray): Scaled input features.
        """
        if self.scaler:
            X_scaled = self.scaler.transform(X_features)
        else:
            X_scaled = X_features
        return X_scaled

    def train(self, X_raw: Sequence[str], y: np.ndarray) -> None:
        """
        Train the sklearn model with the provided data.
        If a scaler has been provided also fit the scaler.

        Args:
            X_raw (Sequence[str]): A list of SMILES strings.
            y (np.ndarray): Target values.

        """
        X_features = self.featurize_data(X_raw)
        if self.scaler:
            self.scaler.fit(X_features)
        X_scaled = self.scale_data(X_features)
        self.model.fit(X_scaled, y)

    def predict(self, X_raw: Sequence[str]) -> np.ndarray:
        """
        Make predictions with the trained model.
        Assumes model has already been trained.

        Args:
            X_raw (Sequence[str]): A list of SMILES strings.

        Returns:
            predictions (np.ndarray): Predicted values.
        """
        X_features = self.featurize_data(X_raw)
        X_scaled = self.scale_data(X_features)
        predictions = self.model.predict(X_scaled)
        return predictions

    def save_model(self, rootpath: Path):
        """
        Save the trained model, featurizer and scaler (if provided) to disk.

        Args:
            rootpath (Path): The directory to save the model, featurizer and scaler into.

        """
        if rootpath.exists() and not rootpath.is_dir():
            raise ValueError(f"{rootpath=} exists and is not a directory.")
        rootpath.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.model, rootpath / "sklearn_model.pkl")
        self.featurizer.to_state_yaml_file(
            rootpath / "featurizer_state_dict.yml"
        )
        if self.scaler:
            joblib.dump(self.scaler, rootpath / "scaler.pkl")
