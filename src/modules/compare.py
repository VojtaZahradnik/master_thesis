import pickle
from src.modules import conf, log, compare_features, evl
import os
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pandas as pd

from io import BytesIO
import base64

class Compare:

    def __init__(self, ref_athlete_name: str):
        self.ref_athlete_name = ref_athlete_name

        self.load_ref_model()

    def load_ref_model(self):
        path = os.path.join(conf["Paths"]["ref_models"], f"{self.ref_athlete_name}.sav")
        if os.path.exists(path):
            self.ref_model = pickle.load(open(path, 'rb'))
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")

    def calc_importances(self, clf: XGBRegressor, cols: list):
        importances_model1 = clf.feature_importances_
        importances_model2 = self.ref_model.feature_importances_

        selected_indices = [list(cols).index(feature) for feature in compare_features]

        self.importances_model1_selected = importances_model1[selected_indices]
        self.importances_model2_selected = importances_model2[selected_indices]


    def plot_radar(self):
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