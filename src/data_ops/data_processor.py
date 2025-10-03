import json
import csv
import pandas as pd
from pathlib import Path
from .data_loader import DataLoader


class DataProcessor:
    """
    Processes raw energy system input data loaded by DataLoader into structured
    formats ready for the optimization model.

    Attributes:
        loader (DataLoader): instance of DataLoader to fetch raw data
        processed_data (dict): dictionary with processed parameters for optimization
    """

    def __init__(self, input_path: Path, question: str):
        """
        Initialize the DataProcessor.

        Args:
            input_path (Path): path to the data folder
            question (str): question name/scenario
        """
        self.loader = DataLoader(input_path=input_path, question=question)
        self.processed_data = {}

    def process(self):
        """
        Load raw data via DataLoader and process it into optimization model parameters.

        Outputs stored in self.processed_data:
            - Pt: list of hourly PV production
            - Dtot: total daily energy demand
            - lambda_t: hourly electricity prices
            - FI: import tariff
            - FE: export tariff
            - PI: max import
            - PE: max export
            - penalty_import_excess
            - penalty_export_excess
        """
        # Load raw JSON and CSV data
        raw_data = self.loader.load_all_json()

        # Extract PV production (assumes one consumer, one PV)
        profile = raw_data["DER_production"][0]["hourly_profile_ratio"]
        max_power = raw_data["appliance_params"]["DER"][0]["max_power_kW"]
        self.processed_data["Pt"] = [ratio * max_power for ratio in profile]


        # Total energy demand from usage preferences
        load_pref = raw_data["usage_preference"][0]["load_preferences"][0]
        self.processed_data["Dtot"] = load_pref["min_total_energy_per_day_hour_equivalent"]

        # Hourly electricity prices
        self.processed_data["lambda_t"] = raw_data["bus_params"][0]["energy_price_DKK_per_kWh"]

        # Hourly max load
        self.processed_data["Dhour"] = raw_data["appliance_params"]["load"][0]["max_load_kWh_per_hour"]


        # Grid tariffs
        self.processed_data["F_I"] = raw_data["bus_params"][0]["import_tariff_DKK/kWh"]
        self.processed_data["F_E"] = raw_data["bus_params"][0]["export_tariff_DKK/kWh"]

        # Max import/export limits
        self.processed_data["P_I"] = raw_data["bus_params"][0]["max_import_kW"]
        self.processed_data["P_E"] = raw_data["bus_params"][0]["max_export_kW"]

        # Optional: penalty for exceeding max import/export
        self.processed_data["penalty_import_excess"] = raw_data["bus_params"][0].get(
            "penalty_excess_import_DKK/kWh", 0
        )
        self.processed_data["penalty_export_excess"] = raw_data["bus_params"][0].get(
            "penalty_excess_export_DKK/kWh", 0
        )
