from pathlib import Path
from src.runner.runner import Runner
from src.opt_model.opt_model import (
    OptModel,
    OptModelFlex,
    OptModelFlexBattery,
    OptModelFlexBatteryInvestment
)
from src import sensitivity

def main():
    # -------------------------
    # Paths
    # -------------------------
    project_root = Path(__file__).parent.resolve()
    question_folder = "question_1a"  # change manually
    data_path = project_root.parent / "data" / question_folder
    results_path = project_root.parent / "results" / question_folder
    results_path.mkdir(exist_ok=True)

    # -------------------------
    # Model choice
    # -------------------------
    model_choice = 1  # 1–4

    if model_choice == 1:
        model_class = OptModel
        model_kwargs = {}

    elif model_choice == 2:
        model_class = OptModelFlex
        model_kwargs = {"alpha": 1}

    elif model_choice == 3:
        model_class = OptModelFlexBattery
        model_kwargs = {"alpha": 2}

    elif model_choice == 4:
        model_class = OptModelFlexBatteryInvestment
        model_kwargs = {"alpha": 2}
        phi = 1000  # Battery investment cost

    else:
        raise ValueError("Invalid model_choice (1–4)")

    print(f"\n>>> Running model: {model_class.__name__} <<<")

    # -------------------------
    # Base run
    # -------------------------
    if model_choice == 4:
        runner = Runner(data_path, results_path, question_folder, model_class, model_kwargs, phi=phi)
    else:
        runner = Runner(data_path, results_path, question_folder, model_class, model_kwargs)

    runner.prepare_data_single_simulation()
    runner.run_single_simulation()
    base_results = runner.results

    # -------------------------
    # Sensitivity Analysis
    # -------------------------
    scenario_results_grouped = {}

    # -------- Model 1 --------
    if model_choice == 1:
        for param in ["F_E", "F_I", "lambda_t"]:
            scenarios = sensitivity.generate_economic_scenarios(runner.data, param_name=param)
            scenario_results_grouped[param] = sensitivity.run_scenarios(scenarios, model_class, **model_kwargs)

    # -------- Model 2 --------
    elif model_choice == 2:
        # Flexibility (dt) scenarios
        flex_scenarios = sensitivity.generate_flexibility_scenarios(runner.data)
        scenario_results_grouped["Flexibility"] = {}
        for name, data in flex_scenarios:
            model = model_class(data, alpha=model_kwargs["alpha"])
            model.run()
            scenario_results_grouped["Flexibility"][name] = model.results

        # Alpha scenarios
        base_alpha = model_kwargs["alpha"]
        alpha_scenarios = sensitivity.generate_alpha_scenarios(runner.data, base_alpha)
        scenario_results_grouped["Alpha"] = {}
        for name, alpha_val in alpha_scenarios:
            model = model_class(runner.data, alpha=alpha_val)
            model.run()
            scenario_results_grouped["Alpha"][name] = model.results

    # -------- Model 3 & 4 --------
    elif model_choice in [3, 4]:
        # Flexibility (dt) scenarios
        flex_scenarios = sensitivity.generate_flexibility_scenarios(runner.data)
        if model_choice == 4:
            scenario_results_grouped["Flexibility"] = sensitivity.run_scenarios(flex_scenarios, model_class, **model_kwargs, phi=phi)
        else:
            scenario_results_grouped["Flexibility"] = sensitivity.run_scenarios(flex_scenarios, model_class, **model_kwargs)

        # Economic scenarios
        for param in ["F_E", "F_I", "lambda_t"]:
            econ_scenarios = sensitivity.generate_economic_scenarios(runner.data, param_name=param)
            if model_choice == 4:
                scenario_results_grouped[param] = sensitivity.run_scenarios(econ_scenarios, model_class, **model_kwargs, phi=phi)
            else:
                scenario_results_grouped[param] = sensitivity.run_scenarios(econ_scenarios, model_class, **model_kwargs)

        # Alpha scenarios
        base_alpha = model_kwargs["alpha"]
        alpha_scenarios = sensitivity.generate_alpha_scenarios(runner.data, base_alpha)
        scenario_results_grouped["Alpha"] = {}
        for name, alpha_val in alpha_scenarios:
            if model_choice == 4:
                model = model_class(runner.data, alpha=alpha_val, phi=phi)
            else:
                model = model_class(runner.data, alpha=alpha_val)
            model.run()
            scenario_results_grouped["Alpha"][name] = model.results

    # -------------------------
    # Plotting
    # -------------------------
    # Base case
    sensitivity.plot_base_case(base_results)
    # Objective sensitivity
    sensitivity.plot_objective_sensitivity(base_results, scenario_results_grouped)
    # PV and consumption
    for param_name, scenario_results in scenario_results_grouped.items():
        plot_k = model_choice == 4 and param_name in ["Flexibility", "F_E", "F_I", "lambda_t", "Alpha"]
        sensitivity.plot_pv_and_consumption_scenarios(base_results, scenario_results, param_name, plot_k=plot_k)

if __name__ == "__main__":
    main()
