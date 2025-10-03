from pathlib import Path
from src.data_ops import DataLoader, DataProcessor, DataVisualizer
from src.opt_model import OptModel
from src.runner.runner import Runner

def main():
    # -------------------------
    # 1. Set paths
    # -------------------------
    project_root = Path(__file__).parent.resolve()  # src folder
    data_path = project_root.parent / "data" / "question_1a"  # folder with your JSON files
    results_path = project_root.parent / "results"

    # -------------------------
    # 2. Initialize runner
    # -------------------------
    runner = Runner(input_path=data_path, output_path=results_path, question="question_1a")

    # -------------------------
    # 3. Prepare data
    # -------------------------
    runner.prepare_data_single_simulation()

    # -------------------------
    # 4. Run optimization
    # -------------------------
    runner.run_single_simulation()

    # -------------------------
    # 5. Save results
    # -------------------------
    runner.save_results("Q1a_results.json")

    # -------------------------
    # 6. Optional: visualize results
    # -------------------------
    visualizer = DataVisualizer()
    # Example: you could later implement something like:
    # visualizer.plot_hourly_power(runner.results["question_1a"])

    print("Q1a simulation finished. Results saved in:", results_path)

if __name__ == "__main__":
    main()
