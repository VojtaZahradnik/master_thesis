import pickle
from src.modules import conf, compare_features
import os
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from io import BytesIO
import base64

class Compare:
    """
     This class provides functionality to compare the feature importances of a reference athlete's model
     with another athlete's model using XGBoost regressors. It also includes a method to visualize
     these comparisons using a radar plot.

     Attributes:
         ref_athlete_name (str): Name of the reference athlete.
         ref_model (pickle.Model): The loaded model for the reference athlete.
         importances_model1_selected (numpy.ndarray): The selected feature importances of the first model.
         importances_model2_selected (numpy.ndarray): The selected feature importances of the reference model.

     Methods:
         __init__(self, ref_athlete_name: str): Constructor that initializes the class with the reference athlete's name.
         load_ref_model(self): Loads the reference model from a specified path.
         calc_importances(self, clf: XGBRegressor, cols: list): Calculates and stores the feature importances for
             both the passed classifier and the reference model.
         plot_radar(self): Generates and returns a base64-encoded string of the radar plot comparing feature importances.
    """

    def __init__(self, ref_athlete_name: str) -> None:
        """
        Initializes the Compare class with the name of the reference athlete. It also calls the method to load
        the reference model associated with the athlete.

        Args:
            ref_athlete_name (str): The name of the reference athlete.
        """
        self.ref_athlete_name = ref_athlete_name

        self.load_ref_model()

    def load_ref_model(self) -> None:
        """
        Loads the reference model from a file. The file path is constructed using the reference athlete's name.
        Raises FileNotFoundError if the model file does not exist.
        """
        path = os.path.join(conf["Paths"]["ref_models"], f"{self.ref_athlete_name}.sav")
        if os.path.exists(path):
            self.ref_model = pickle.load(open(path, 'rb'))
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")

    def calc_importances(self, clf: XGBRegressor, cols: list) -> None:
        """
        Calculates the feature importances for both the provided classifier and the reference model.
        Selected features are determined by the global 'compare_features' list.

        Args:
            clf (XGBRegressor): The XGBoost regressor model for comparison.
            cols (list): A list of column names used in the model.
        """
        importances_model1 = clf.feature_importances_
        importances_model2 = self.ref_model.feature_importances_

        selected_indices = [list(cols).index(feature) for feature in compare_features]

        self.importances_model1_selected = importances_model1[selected_indices]
        self.importances_model2_selected = importances_model2[selected_indices]


    def plot_radar(self) -> None:
        """
        Generates a radar plot to visually compare the feature importances of the two models.
        The plot is saved in memory and encoded in base64 format for easy sharing or embedding.

        Returns:
            plot (str): A base64-encoded string of the radar plot.
        """
        # Angle of each axis in the plot
        angles = np.linspace(0, 2 * np.pi, len(compare_features), endpoint=False)
        # Make the plot circular
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot each model's feature importances
        ax.plot(angles, self.importances_model1_selected, label=conf["Athlete"]["name"], linewidth=2, linestyle='solid')
        ax.fill(angles, self.importances_model1_selected, alpha=0.25)

        ax.plot(angles, self.importances_model2_selected, label='Reference Athlete', linewidth=2, linestyle='solid')
        ax.fill(angles, self.importances_model2_selected, alpha=0.25)

        # Set labels for each axis
        ax.set_thetagrids(angles * 180 / np.pi, compare_features)

        # Add a legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Show the plot
        plt.title('Feature Importance Comparison')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()

        return plot