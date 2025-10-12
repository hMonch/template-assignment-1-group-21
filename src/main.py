from pathlib import Path
import numpy as np
from src.runner.runner import Runner
from src.opt_model.opt_model import (
    OptModel,
    OptModelFlex,
    OptModelFlexBattery,
    OptModelFlexBatteryInvestment
)
from src import sensitivity
from src.utils.utils import create_demand_profiles

def main():
    # -------------------------
    # Paths
    # -------------------------
    project_root = Path(__file__).parent.resolve()
    question_folder = "question_1c"  # change manually
    data_path = project_root.parent / "data" / question_folder
    results_path = project_root.parent / "results" / question_folder
    results_path.mkdir(exist_ok=True)

    # -------------------------
    # Model choice
    # -------------------------
    model_choice = 4 # 1–4

    if model_choice == 1:
        model_class = OptModel
        model_kwargs = {}

    elif model_choice == 2:
        model_class = OptModelFlex
        model_kwargs = {"alpha": 1.5}

    elif model_choice == 3:
        model_class = OptModelFlexBattery
        model_kwargs = {"alpha": 1.5}

    elif model_choice == 4:
        model_class = OptModelFlexBatteryInvestment
        model_kwargs = {"alpha": 1.5}
        phi = 5000  #DKK/kWh Battery investment cost

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
        
        #Demand scenarios
        d_profiles = create_demand_profiles(['industrial', 'office'])
        demand_scenarios = sensitivity.generate_demand_profiles_scenarios(runner.data, demand_profiles=d_profiles)
        scenario_results_grouped["Demand type"] = {}
        for name, data in demand_scenarios:
            model = model_class(data, alpha=model_kwargs["alpha"])
            model.run()
            scenario_results_grouped["Demand type"][name] = model.results

    # -------- Model 3 & 4 --------
    elif model_choice in [3, 4]:
        #Question 1.c, main flexibility analysis
        merge_scenario_results = {}

        alpha_list = [0.5,1.5,3]

        d_profiles = create_demand_profiles(['industrial', 'office'])

        model_2 = OptModelFlex
        model_3 = OptModelFlexBattery
        for alpha in alpha_list:
            merge_scenario_results[alpha] = {}

            demand_scenarios = sensitivity.generate_demand_profiles_scenarios(runner.data, demand_profiles=d_profiles)

            for name, data in demand_scenarios:
                merge_scenario_results[alpha][name] = {'No_Battery': None, 'Battery': None}
                model2 = model_2(data, alpha)
                model3 = model_3(data, alpha)
                model2.run()
                model3.run()

                merge_scenario_results[alpha][name]["No_Battery"] = model2.results
                merge_scenario_results[alpha][name]["Battery"] = model3.results
        
        sensitivity.plot_1c_objfunc_sensitivity_demand(merge_scenario_results)

         #Question 2.b Demand scenarios
        scenario_results_demand = {}
        if model_choice == 4:
             d_profiles = create_demand_profiles(['industrial', 'office'])
             demand_scenarios = sensitivity.generate_demand_profiles_scenarios(runner.data, demand_profiles=d_profiles)
             scenario_results_demand["Demand type"] = {}
             factor = [0.5,1,2,4,8]

             for name, data in demand_scenarios:
                 if name not in scenario_results_demand["Demand type"]:
                     scenario_results_demand["Demand type"][name] = {}

                 flexibility_scenarios = sensitivity.generate_flexibility_scenarios(data, factor)

                 for scenario_name, scaled_data in flexibility_scenarios:
                     f = float(scenario_name.split("_")[-1])

                     print('charge totale:',np.sum(scaled_data["dt"]))

                     model = model_class(scaled_data, alpha=1000, phi=2500)
                     model.run()

                     scenario_results_demand["Demand type"][name][f] = model.results

             sensitivity.plotk_2b(scenario_results_demand, runner.data, factor)
        #------------------------------------------------------------------------------#


        base_alpha = model_kwargs["alpha"]
        alpha_scenarios = sensitivity.generate_alpha_scenarios(runner.data, base_alpha, alpha_list=[0.5, 1.5, 3])
        scenario_results_grouped["AlphaB"] = {}
        model_2 = OptModelFlex
        model_3 = OptModelFlexBattery
        for name, alpha_val in alpha_scenarios:
            model2 = model_2(runner.data, alpha_val)
            model3 = model_3(runner.data, alpha_val)
            model2.run()
            model3.run()

            if alpha_val not in scenario_results_grouped["AlphaB"]:
                scenario_results_grouped["AlphaB"][alpha_val] = {}
            
            scenario_results_grouped["AlphaB"][alpha_val]["No_Battery"] = model2.results
            scenario_results_grouped["AlphaB"][alpha_val]["Battery"] = model3.results
        #We plot the graphs of interests
        sensitivity.plot_1c_objfunc_sensitivity(scenario_results_grouped)

        #--------------------------------------------------------------------------------------------#

        #We reinitialize the dic with scenario results
        scenario_results_grouped={}



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
    sensitivity.plot_base_case(base_results, runner.data, model_choice)
    # Objective sensitivity
    sensitivity.plot_objective_sensitivity(base_results, scenario_results_grouped)
    # PV and consumption
    for param_name, scenario_results in scenario_results_grouped.items():
        if param_name == "Demand type":
            sensitivity.plot_comparison_demand_consumption(runner.data, d_profiles, base_results, scenario_results, param_name)
        plot_k = model_choice == 4 and param_name in ["Flexibility", "F_E", "F_I", "lambda_t", "Alpha", "Demand type"]
        #sensitivity.plot_pv_and_consumption_scenarios(base_results, runner.data, scenario_results, param_name, plot_k=plot_k)

if __name__ == "__main__":
    main()
