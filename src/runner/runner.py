from pathlib import Path
from typing import List, Type, Optional, Dict

from src.data_ops import DataProcessor
from src.opt_model.opt_model import OptModel, OptModelFlex, OptModelFlexBattery 


class Runner:
    """
    Handles configuration, data loading, model execution, and storing results.
    Can run either OptModel or OptModelFlex based on user choice.
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        question: str,
        model_class: Type = OptModelFlex,
        model_kwargs: Optional[Dict] = None
    ) -> None:
        """
        Args:
            input_path (Path): Path to input data folder
            output_path (Path): Path to store outputs
            question (str): Question/scenario to run
            model_class (Type): Either OptModel or OptModelFlex
            model_kwargs (dict, optional): Extra keyword arguments to pass to the model (e.g., alpha)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.question = question
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}

        self.data = None
        self.results = None
        self.all_data = {}

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

        # Instantiate the chosen model with any kwargs (e.g., alpha)
        self.model = self.model_class(self.data, **self.model_kwargs)
        self.model.run()
        self.model.display_results()

        # Save results for later use (e.g., plotting)
        self.results = self.model.results

    def run_all_simulations(self) -> None:
        """Run all simulations for multiple questions or scenarios."""
        if not self.all_data:
            raise ValueError("All data not prepared. Run prepare_data_all_simulations() first.")

        self.results = {}  # reset results dictionary
        for q, data in self.all_data.items():
            print(f"\n--- Running simulation for question: {q} ---")
            model = self.model_class(data, **self.model_kwargs)
            model.run()
            model.display_results()
            self.results[q] = model.results
