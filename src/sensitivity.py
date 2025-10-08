import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from src.opt_model.opt_model import OptModel, OptModelFlex, OptModelFlexBattery

# -------------------------
# Scenario generators
# -------------------------

def generate_economic_scenarios(base_data, param_name, n_scenarios=4, increase_pct=0.5):
    scenarios = []
    for i in range(1, n_scenarios + 1):
        factor = 1 + increase_pct * i
        data_scenario = deepcopy(base_data)

        if param_name == "lambda_t":
            data_scenario["lambda_t"] = [x * factor for x in base_data["lambda_t"]]
        else:
            data_scenario[param_name] = base_data[param_name] * factor

        scenarios.append((f"{param_name}_scenario_{i}", data_scenario))
    return scenarios


def generate_flexibility_scenarios(base_data, n_scenarios=8, factor_profile=1.4, factor_maxload=1/1.7):
    scenarios = []
    half = n_scenarios // 2

    # Scale dt
    for i in range(1, half + 1):
        data_scenario = deepcopy(base_data)
        factor = 1 + (factor_profile - 1) * i / half
        data_scenario["dt"] = [x * factor for x in data_scenario["dt"]]
        scenarios.append((f"dt_factor_{factor:.3f}", data_scenario))

    # Scale Dhour
    for i in range(1, half + 1):
        data_scenario = deepcopy(base_data)
        factor = 1 - (1 - factor_maxload) * i / half
        data_scenario["Dhour"] = data_scenario["Dhour"] * factor
        scenarios.append((f"Dh_max_{data_scenario['Dhour']:.2f}", data_scenario))

    return scenarios


# -------------------------
# Run scenarios
# -------------------------

def run_scenarios(scenarios, model_class, alpha=None):
    results = {}
    for name, data in scenarios:
        if model_class in [OptModelFlex, OptModelFlexBattery]:
            model = model_class(data, alpha)
        else:
            model = model_class(data)
        model.run()
        results[name] = deepcopy(model.results)
    return results


# -------------------------
# Helper: pretty labels
# -------------------------

def format_label(name):
    """Create readable labels for scenarios."""
    if "F_E" in name:
        idx = int(name.split("_")[-1])
        factor = 1 + 0.5 * idx
        return f"Export fee factor = {factor:.2f}"
    elif "F_I" in name:
        idx = int(name.split("_")[-1])
        factor = 1 + 0.5 * idx
        return f"Import fee factor = {factor:.2f}"
    elif "lambda_t" in name:
        idx = int(name.split("_")[-1])
        factor = 1 + 0.5 * idx
        return f"Electricity price factor = {factor:.2f}"
    elif "dt_factor" in name:
        factor = float(name.split("_")[-1])
        return f"Reference load factor = {factor:.2f}"
    elif "Dh_max" in name:
        value = float(name.split("_")[-1])
        return f"Max load = {value:.2f} kWh"
    else:
        return name


def format_group_title(param_name):
    """Convert scenario group to a human-readable title."""
    if param_name == "F_E":
        return "Export Fee Sensitivity"
    elif param_name == "F_I":
        return "Import Fee Sensitivity"
    elif param_name == "lambda_t":
        return "Electricity Price Sensitivity"
    elif param_name == "Flexibility":
        return "Flexibility Sensitivity"
    else:
        return param_name


# -------------------------
# Plotting functions
# -------------------------

def plot_base_case(base_results):
    plt.figure(figsize=(12, 6))
    plt.plot(base_results["pt"], label="PV Production")
    plt.plot(base_results["pI"], label="Grid Import")
    plt.plot(base_results["pE"], label="Grid Export")
    plt.title("Base Case: Hourly PV, Import, Export")
    plt.xlabel("Hour")
    plt.ylabel("Power [kW]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scenarios(base_results, scenario_results, scenario_name, show_net_consumption=False):
    """
    Plot scenario results.
    If show_net_consumption=True, only plot total consumption (pt + pI - pE)
    """
    plt.figure(figsize=(12, 6))
    
    base_consumption = [base_results["pt"][t] + base_results["pI"][t] - base_results["pE"][t] for t in range(24)]
    
    if show_net_consumption:
        plt.plot(base_consumption, '--', color='black', label='Base Consumption')
        for name, res in scenario_results.items():
            scenario_consumption = [res["pt"][t] + res["pI"][t] - res["pE"][t] for t in range(24)]
            plt.plot(scenario_consumption, label=format_label(name))
        plt.ylabel("Consumption [kWh]")
        plt.title(f'{format_group_title(scenario_name)}: Net Consumption')
    else:
        plt.plot(base_results["pt"], '--', label="Base PV")
        plt.plot(base_results["pI"], '--', label="Base Import")
        plt.plot(base_results["pE"], '--', label="Base Export")
        for name, res in scenario_results.items():
            plt.plot(res["pt"], label=f"{format_label(name)} PV")
            plt.plot(res["pI"], label=f"{format_label(name)} Import")
            plt.plot(res["pE"], label=f"{format_label(name)} Export")
        plt.ylabel("Power [kW]")
        plt.title(f"{format_group_title(scenario_name)}")
    
    plt.xlabel("Hour")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_objective_sensitivity(base_results, scenario_results_grouped):
    """
    Plot the total cost (objective) sensitivity for all scenario groups.
    """
    for param, scenarios in scenario_results_grouped.items():
        scenario_names = [format_label(name) for name in scenarios.keys()]
        objectives = [res["objective_value"] for res in scenarios.values()]

        plt.figure(figsize=(8, 5))
        plt.bar(scenario_names, objectives, color="skyblue", label="Scenarios")
        plt.axhline(y=base_results["objective_value"], color="red", linestyle="--", label="Base Case")
        plt.title(f"{format_group_title(param)}: Objective Value Sensitivity")
        plt.ylabel("Total Cost [DKK]")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()
