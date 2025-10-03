import json
from pathlib import Path
import numpy as np

class DataLoader:
    """
    Loads energy system input data from structured JSON files for a given question/scenario.

    Attributes:
        input_path (Path): Path to the main folder containing 'data' (e.g., template-assignment-1-group-21).
        question (str): Question/scenario name (e.g., "question_1a").
        data (dict): Dictionary storing all loaded JSON files by filename stem.
    """

    def __init__(self, input_path: str | Path, question: str):
        self.input_path = Path(input_path).resolve()  # main folder containing 'data'
        self.question = question
        self.data = {}

    def load_all_json(self):
       """
         Load all JSON files in the specified question folder into self.data.
     """
       question_path = self.input_path  # remove extra "data"
       if not question_path.exists():
         raise FileNotFoundError(f"Question folder not found: {question_path}")

    # Load all JSON files
       for file_path in question_path.glob("*.json"):
         with open(file_path, "r", encoding="utf-8") as f:
            self.data[file_path.stem] = json.load(f)
       return self.data


    def get_consumer(self):
        """
        Returns the first consumer from consumer_params.json.
        """
        consumers = self.data.get("consumer_params")
        if not consumers:
            raise ValueError("consumer_params.json not loaded")
        return consumers[0]

    def get_bus(self, bus_id: str):
        """
        Returns the bus dictionary matching the given bus_id.
        """
        bus_list = self.data.get("bus_params")
        if not bus_list:
            raise ValueError("bus_params.json not loaded")
        bus = next((b for b in bus_list if b["bus_ID"] == bus_id), None)
        if bus is None:
            raise ValueError(f"Bus {bus_id} not found in bus_params.json")
        return bus

    def get_pv_profile(self):
        """
        Returns the PV hourly production profile for the first DER in DER_production.json.
        """
        der_list = self.data.get("DER_production")
        if not der_list:
            raise ValueError("DER_production.json not loaded")
        return der_list[0]["hourly_profile_ratio"]

