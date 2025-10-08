import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from src.opt_model.opt_model import (
    OptModel,
    OptModelFlex,
    OptModelFlexBattery,
    OptModelFlexBatteryInvestment
)

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

def generate_flexibility_scenarios(base_data, n_scenarios=4, factor_profile=1.4):
    """Generate scenarios by scaling dt only (max load removed)."""
    scenarios = []
    for i in range(1, n_scenarios + 1):
        factor = 1 + (factor_profile - 1) * i / n_scenarios
        data_scenario = deepcopy(base_data)
        data_scenario["dt"] = [x * factor for x in base_data["dt"]]
        scenarios.append((f"dt_factor_{factor:.3f}", data_scenario))
    return scenarios

def generate_alpha_scenarios(base_data, base_alpha, n_scenarios=4, increase_pct=0.5):
    """Generate alpha values increasing from base_alpha."""
    scenarios = []
    for i in range(1, n_scenarios + 1):
        alpha_val = base_alpha * (1 + increase_pct * i)
        scenarios.append((f"alpha_{alpha_val:.2f}", alpha_val))
    return scenarios

# -------------------------
# Run scenarios
# -------------------------

def run_scenarios(scenarios, model_class, alpha=None, phi=None):
    results = {}
    for name, data in scenarios:
        if model_class is OptModelFlexBatteryInvestment:
            model = model_class(data, alpha=alpha, phi=phi)
        elif model_class in [OptModelFlex, OptModelFlexBattery]:
            model = model_class(data, alpha=alpha)
        else:
            model = model_class(data)

        model.run()
        results[name] = deepcopy(model.results)
    return results

# -------------------------
# Helpers
# -------------------------

def format_label(name):
    if "F_E" in name:
        idx = int(name.split("_")[-1])
        factor = 1 + 0.5 * idx
        return f"Export fee ×{factor:.2f}"
    elif "F_I" in name:
        idx = int(name.split("_")[-1])
        factor = 1 + 0.5 * idx
        return f"Import fee ×{factor:.2f}"
    elif "lambda_t" in name:
        idx = int(name.split("_")[-1])
        factor = 1 + 0.5 * idx
        return f"Price ×{factor:.2f}"
    elif "alpha" in name:
        val = float(name.split("_")[-1])
        return f"α = {val:.1f}"
    elif "dt_factor" in name:
        factor = float(name.split("_")[-1])
        return f"Load factor = {factor:.2f}"
    else:
        return name

def format_group_title(param_name):
    mapping = {
        "F_E": "Export Fee Sensitivity",
        "F_I": "Import Fee Sensitivity",
        "lambda_t": "Electricity Price Sensitivity",
        "Flexibility": "Flexibility Sensitivity",
        "Alpha": "Alpha Sensitivity",
    }
    return mapping.get(param_name, param_name)

# -------------------------
# Plotting
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

def plot_pv_and_consumption_scenarios(base_results, scenario_results, scenario_name, plot_k=False):
    # PV Production
    plt.figure(figsize=(10, 5))
    plt.plot(base_results["pt"], "--", color="black", label="Base PV")
    for name, res in scenario_results.items():
        plt.plot(res["pt"], label=format_label(name))
    plt.title(f"{format_group_title(scenario_name)}: PV Production")
    plt.xlabel("Hour")
    plt.ylabel("PV Power [kW]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Net consumption
    plt.figure(figsize=(10, 5))
    base_cons = [base_results["pt"][t] + base_results["pI"][t] - base_results["pE"][t] for t in range(24)]
    plt.plot(base_cons, "--", color="black", label="Base Consumption")
    for name, res in scenario_results.items():
        cons = [res["pt"][t] + res["pI"][t] - res["pE"][t] for t in range(24)]
        plt.plot(cons, label=format_label(name))
    plt.title(f"{format_group_title(scenario_name)}: Net Consumption")
    plt.xlabel("Hour")
    plt.ylabel("Energy [kWh]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot battery k if requested
    if plot_k:
        plt.figure(figsize=(8, 5))
        ks = [res["k"] for res in scenario_results.values()]
        labels = [format_label(name) for name in scenario_results.keys()]
        plt.bar(labels, ks, color="orange")
        plt.title(f"{format_group_title(scenario_name)}: Battery Size k")
        plt.ylabel("k [kWh]")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()

def plot_objective_sensitivity(base_results, scenario_results_grouped):
    for param, scenarios in scenario_results_grouped.items():
        scenario_names = [format_label(name) for name in scenarios.keys()]
        objectives = [res["objective_value"] for res in scenarios.values()]
        plt.figure(figsize=(8, 5))
        plt.bar(scenario_names, objectives, color="skyblue")
        plt.axhline(y=base_results["objective_value"], color="red", linestyle="--", label="Base Case")
        plt.title(f"{format_group_title(param)}: Objective Sensitivity")
        plt.ylabel("Total Cost [DKK]")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()
