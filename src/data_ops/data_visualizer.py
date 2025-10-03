import json
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

class DataVisualizer:
    """Class to visualize optimization results for Q1a."""

    def __init__(self):
        pass

    def plot_pv_import_export(self, results):
        """
        Plot PV production, import, and export over 24 hours.

        results: dictionary containing arrays
            - pt: PV production per hour [kW]
            - pI: imported power per hour [kW]
            - pE: exported power per hour [kW]
        """
        hours = range(24)
        plt.figure(figsize=(10, 6))
        plt.plot(hours, results["pt"], label="PV production", marker='o')
        plt.plot(hours, results["pI"], label="Grid import", marker='o')
        plt.plot(hours, results["pE"], label="Grid export", marker='o')
        plt.xlabel("Hour")
        plt.ylabel("Power [kW]")
        plt.title("PV Production and Grid Interaction Over 24 Hours")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_costs(self, results):
        """
        Plot hourly net cost using import/export and price.

        results: dictionary containing arrays
            - pt: PV production per hour [kW]
            - pI: imported power per hour [kW]
            - pE: exported power per hour [kW]
            - lambda_t: electricity price per hour [DKK/kWh]
            - F_I: import tariff
            - F_E: export tariff
        """
        hours = range(24)
        cost = results["pI"] * (results["lambda_t"] + results["F_I"]) - \
               results["pE"] * (results["lambda_t"] - results["F_E"])

        plt.figure(figsize=(10, 6))
        plt.bar(hours, cost, color='skyblue')
        plt.xlabel("Hour")
        plt.ylabel("Cost [DKK]")
        plt.title("Hourly Net Energy Cost")
        plt.grid(True)
        plt.show()

    def plot_prices(self, lambda_t):
        """
        Plot hourly electricity price over 24 hours.

        lambda_t: array-like of 24 values [DKK/kWh]
        """
        hours = range(24)
        plt.figure(figsize=(10, 6))
        plt.plot(hours, lambda_t, marker='o', color='orange')
        plt.xlabel("Hour")
        plt.ylabel("Electricity Price [DKK/kWh]")
        plt.title("Hourly Electricity Price")
        plt.grid(True)
        plt.show()
