from pathlib import Path
from typing import List

from src.data_ops import DataProcessor
from src.opt_model import OptModel


class Runner:
    """
    Handles configuration setting, data loading and preparation, model(s) execution, results printing.
    """

    def __init__(self, input_path: Path, output_path: Path, question: str) -> None:
        """Initialize the Runner with paths and scenario/question."""
        self.input_path = input_path
        self.output_path = output_path
        self.question = question
        self.data = None
        self.results = None

    def prepare_data_single_simulation(self) -> None:
        """Prepare input data for a single simulation using DataProcessor."""
        processor = DataProcessor(input_path=self.input_path, question=self.question)
        processor.process()
        self.data = processor.processed_data

    def prepare_data_all_simulations(self, question_list: List[str]) -> None:
        """Prepare input data for multiple scenarios/questions."""
        self.all_data = {}
        for q in question_list:
            processor = DataProcessor(input_path=self.input_path, question=q)
            processor.process()
            self.all_data[q] = processor.processed_data

    def run_single_simulation(self) -> None:
        """Run a single simulation for the loaded data."""
        if self.data is None:
            raise ValueError("Data not prepared. Run prepare_data_single_simulation() first.")

        # Pass data as a positional argument, not as a keyword
        self.model = OptModel(self.data)
        self.model.run()
        self.model.display_results()

    def run_all_simulations(self) -> None:
        """Run all simulations for multiple questions."""
        if not hasattr(self, "all_data"):
            raise ValueError("All data not prepared. Run prepare_data_all_simulations() first.")

        for q, data in self.all_data.items():
            print(f"\n--- Running simulation for question: {q} ---")
            model = OptModel(data)
            model.run()
            model.display_results()
