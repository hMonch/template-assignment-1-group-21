from pathlib import Path
from src.runner.runner import Runner
from src.opt_model.opt_model import OptModel, OptModelFlex, OptModelFlexBattery
from src import sensitivity


def main():
    # -------------------------
    # Paths (YOU choose question folder manually)
    # -------------------------
    project_root = Path(__file__).parent.resolve()

    # Change manually
    question_folder = "question_1a"  # <-- change this manually
    data_path = project_root.parent / "data" / question_folder
    results_path = project_root.parent / "results" / question_folder
    results_path.mkdir(exist_ok=True)

    # -------------------------
    # Model choice
    # -------------------------
    model_choice = 1  # <-- change this manually

    if model_choice == 1:
        model_class = OptModel
        model_kwargs = {}
    elif model_choice == 2:
        model_class = OptModelFlex
        model_kwargs = {"alpha": 2}
    elif model_choice == 3:
        model_class = OptModelFlexBattery
        model_kwargs = {"alpha": 2}
    else:
        raise ValueError("Invalid model_choice. Choose 1, 2, or 3.")

    print(f"\n>>> Running model: {model_class.__name__} <<<")

    # -------------------------
    # Run base case
    # -------------------------
    runner = Runner(
        input_path=data_path,
        output_path=results_path,
        question=question_folder,
        model_class=model_class,
        model_kwargs=model_kwargs
    )

    runner.prepare_data_single_simulation()
    runner.run_single_simulation()
    base_results = runner.results

    # -------------------------
    # Sensitivity analysis
    # -------------------------
    scenario_results_grouped = {}

    if model_choice == 1:
        # Economic only
        econ_params = ["F_E", "F_I", "lambda_t"]
        for param in econ_params:
            scenarios = sensitivity.generate_economic_scenarios(runner.data, param_name=param)
            scenario_results_grouped[param] = sensitivity.run_scenarios(scenarios, model_class, **model_kwargs)

    elif model_choice == 2:
        # Flexibility only
        scenarios = sensitivity.generate_flexibility_scenarios(runner.data)
        scenario_results_grouped["Flexibility"] = sensitivity.run_scenarios(scenarios, model_class, **model_kwargs)

    elif model_choice == 3:
        # Both flexibility and economic
        scenarios_flex = sensitivity.generate_flexibility_scenarios(runner.data)
        scenario_results_grouped["Flexibility"] = sensitivity.run_scenarios(scenarios_flex, model_class, **model_kwargs)

        econ_params = ["F_E", "F_I", "lambda_t"]
        for param in econ_params:
            scenarios_econ = sensitivity.generate_economic_scenarios(runner.data, param_name=param)
            scenario_results_grouped[param] = sensitivity.run_scenarios(scenarios_econ, model_class, **model_kwargs)

    # -------------------------
    # Plotting
    # -------------------------
    # Base case PV/import/export
    sensitivity.plot_base_case(base_results)

    # Objective function sensitivity
    sensitivity.plot_objective_sensitivity(base_results, scenario_results_grouped)

    # Scenario-wise plots
    for param_name, scenario_results in scenario_results_grouped.items():
        # Economic scenarios -> net consumption only
        if param_name in ["F_E", "F_I", "lambda_t"]:
            sensitivity.plot_scenarios(base_results, scenario_results, scenario_name=param_name, show_net_consumption=True)
        else:
            # Flexibility scenarios -> keep original PV/import/export plots
            sensitivity.plot_scenarios(base_results, scenario_results, scenario_name=param_name, show_net_consumption=False)


if __name__ == "__main__":
    main()
