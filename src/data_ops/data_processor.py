import json
import csv
import pandas as pd
from pathlib import Path
from .data_loader import DataLoader


class DataProcessor:
    """
    Processes raw energy system input data loaded by DataLoader into structured
    formats ready for the optimization model.
    """

    def __init__(self, input_path: Path, question: str):
        self.loader = DataLoader(input_path=input_path, question=question)
        self.processed_data = {}

    def process(self):
        """
        Process all data into model-ready inputs.
        """
        # Load raw JSON and CSV data
        raw_data = self.loader.load_all_json()

        # Extract PV production (assumes one consumer, one PV)
        profile = raw_data["DER_production"][0]["hourly_profile_ratio"]
        max_power_pv = raw_data["appliance_params"]["DER"][0]["max_power_kW"]
        self.processed_data["Pt"] = [ratio * max_power_pv for ratio in profile]

        # Total energy demand
        load_pref = raw_data["usage_preferences"][0]["load_preferences"][0]
        max_load = raw_data["appliance_params"]["load"][0]["max_load_kWh_per_hour"]

        # Handle missing minimum energy
        min_energy = load_pref.get("min_total_energy_per_day_hour_equivalent")
        if min_energy is None:
            min_energy = 1e-6  # very small default value
        self.processed_data["Dtot"] = min_energy * max_load

        # Hourly electricity prices
        self.processed_data["lambda_t"] = raw_data["bus_params"][0]["energy_price_DKK_per_kWh"]

        # Hourly max load
        self.processed_data["Dhour"] = max_load

        # Grid tariffs
        self.processed_data["F_I"] = raw_data["bus_params"][0]["import_tariff_DKK/kWh"]
        self.processed_data["F_E"] = raw_data["bus_params"][0]["export_tariff_DKK/kWh"]

        # Max import/export limits
        self.processed_data["P_I"] = raw_data["bus_params"][0]["max_import_kW"]
        self.processed_data["P_E"] = raw_data["bus_params"][0]["max_export_kW"]

        # Optional penalties
        self.processed_data["penalty_import_excess"] = raw_data["bus_params"][0].get(
            "penalty_excess_import_DKK/kWh", 0
        )
        self.processed_data["penalty_export_excess"] = raw_data["bus_params"][0].get(
            "penalty_excess_export_DKK/kWh", 0
        )

        # --- Reference hourly load (dt) for flexible optimization ---
        ref_profile = load_pref.get("hourly_profile_ratio")
        if ref_profile is None:
            ref_profile = [1 / 24] * 24  # equally distributed over 24 hours
        self.processed_data["dt"] = [ratio * max_load for ratio in ref_profile]
        # ------------------------------------------------------------------

        # --- Battery parameters (for scenarios with storage) ---
        storage_pref_list = raw_data["usage_preferences"][0].get("storage_preferences")
        if storage_pref_list:  # not None and not empty
            storage_pref = storage_pref_list[0]
            storage_params = raw_data["appliance_params"]["storage"][0]

            # Basic battery parameters
            C = storage_params["storage_capacity_kWh"]
            eta_ch = storage_params["charging_efficiency"]
            eta_dis = storage_params["discharging_efficiency"]

            # Max charge/discharge power (calculated from ratios × capacity)
            P_ch = storage_params["max_charging_power_ratio"] * C
            P_dis = storage_params["max_discharging_power_ratio"] * C

            # Initial/final SOC ratios from usage preferences
            init_soc_ratio = storage_pref.get("initial_soc_ratio", 0.5)
            final_soc_ratio = storage_pref.get("final_soc_ratio", 0.5)

            # Store in processed data
            self.processed_data.update({
                "C": C,
                "eta_ch": eta_ch,
                "eta_dis": eta_dis,
                "P_ch": P_ch,
                "P_dis": P_dis,
                "initial_soc_ratio": init_soc_ratio,
                "final_soc_ratio": final_soc_ratio,
            })
        else:
            # No storage defined for this scenario, set defaults or None
            self.processed_data.update({
                "C": None,
                "eta_ch": None,
                "eta_dis": None,
                "P_ch": None,
                "P_dis": None,
                "initial_soc_ratio": None,
                "final_soc_ratio": None,
            })
        # -------------------------------------------------------------

        return self.processed_data
