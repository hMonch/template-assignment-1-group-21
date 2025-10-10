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

def generate_economic_scenarios(base_data, param_name
                                #, n_scenarios=4, increase_pct=0.5
                                ):
    scenarios = []
    if param_name == "lambda_t":
        #List of multiplying factor for lambda
        lambda_factor = [0.5,0.75,1,1.5,2,3]
        for i in range(len(lambda_factor)):
            data_scenario = deepcopy(base_data)
            data_scenario["lambda_t"] = [x * lambda_factor[i] for x in base_data["lambda_t"]]
            scenarios.append((f"{param_name}_scenario_{lambda_factor[i]}", data_scenario))

    else:
        tariff_factor = [0,0.5,1,1.5,2,3]
        for i in range(len(tariff_factor)):
            data_scenario = deepcopy(base_data)
            data_scenario[param_name] = base_data[param_name] * tariff_factor[i]
            scenarios.append((f"{param_name}_scenario_{tariff_factor[i]}", data_scenario))


    # # for i in range(1, n_scenarios + 1):
    # #     factor = 1 + increase_pct * i
    # #     data_scenario = deepcopy(base_data)

    # #     if param_name == "lambda_t":
    # #         data_scenario["lambda_t"] = [x * factor for x in base_data["lambda_t"]]
    # #     else:
    # #         data_scenario[param_name] = base_data[param_name] * factor

    # #     scenarios.append((f"{param_name}_scenario_{i}", data_scenario))
    return scenarios


# We generate scenarios for different demand profiles
def generate_demand_profiles_scenarios(base_data, demand_profiles):
    scenarios = []
    for type, demand in demand_profiles.items():
        data_scenario = deepcopy(base_data)
        data_scenario["dt"] = demand
        scenarios.append((f"demand_{type}", data_scenario))
    
    scenarios.append(('demand_base', base_data))

    return scenarios
    

def generate_flexibility_scenarios(base_data, n_scenarios=4, factor_profile=2.5):
    """Generate scenarios by scaling dt only (max load removed)."""
    scenarios = []
    for i in range(1, n_scenarios + 1):
        factor = 1 + (factor_profile - 1) * i / n_scenarios
        data_scenario = deepcopy(base_data)
        data_scenario["dt"] = [x * factor for x in base_data["dt"]]
        scenarios.append((f"dt_factor_{factor:.3f}", data_scenario))
    return scenarios

def generate_alpha_scenarios(base_data, base_alpha, alpha_list = None
                             #, n_scenarios=4, increase_pct=1
                             ):
    """Generate alpha values increasing from base_alpha."""
    scenarios = []
    if alpha_list == None:
        alpha_vals = [0,0.5,1,1.5,2,5,50]
    else:
        alpha_vals = alpha_list
    for i in range(len(alpha_vals)):
        alpha_val = alpha_vals[i]
        scenarios.append((f"alpha_{alpha_val:.2f}", alpha_val))
    # for i in range(1, n_scenarios + 1):
    #     alpha_val = base_alpha * (1 + increase_pct * i)
    #     scenarios.append((f"alpha_{alpha_val:.2f}", alpha_val))
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
        idx = float(name.split("_")[-1])
        factor = idx #1 + 0.5 * idx
        return f"Export fee ×{factor:.1f}"
    elif "F_I" in name:
        idx = float(name.split("_")[-1])
        factor = idx #1 + 0.5 * idx
        return f"Import fee ×{factor:.1f}"
    elif "lambda_t" in name:
        idx = float(name.split("_")[-1])
        factor = idx #1 + 0.5 * idx
        return f"Price ×{factor:.1f}"
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

def plot_comparison_demand_consumption(base_data, scenario_data, base_results, scenario_results, scenario_name):
    #Plot ideal demand vs real consumption
    plt.figure(figsize=(10, 5))
    # base_demand = base_data["dt"]
    # base_cons = [base_results["pt"][t] + base_results["pI"][t] - base_results["pE"][t] for t in range(24)]
    # plt.plot(base_demand, "--", color="black", label="Base PV")
    # plt.plot(base_cons, color="black", label="Base PV")

    colors = ['blue', 'green', 'red']
    for i, (name, res) in enumerate(scenario_results.items()):
        cons = [res["pt"][t] + res["pI"][t] - res["pE"][t] for t in range(24)]
        n = str(name.split("_")[-1])
        if n == 'base':
            demand = base_data['dt']
        else:
            demand = scenario_data[n]

        color = colors[i%len(colors)]
        plt.plot(cons, color = color, label=format_label(n)+' consumption')
        plt.plot(demand, '--', color = color, label=format_label(name))
    plt.title(f"{format_group_title(scenario_name)}: Ideal demand and Net Consumption")
    plt.xlabel("Hour")
    plt.ylabel("Energy [kWh]")
    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
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

    # --- NEW PLOTS BELOW ---

    # Self-sufficiency over time = (consumption - import) / consumption
    plt.figure(figsize=(10, 5))
    base_self_suff = []
    for t in range(24):
        cons = base_results["pt"][t] + base_results["pI"][t] - base_results["pE"][t]
        val = (cons - base_results["pI"][t]) / cons if cons > 0 else 0
        base_self_suff.append(val)
    plt.plot(base_self_suff, "--", color="black", label="Base Self-Sufficiency")

    for name, res in scenario_results.items():
        vals = []
        for t in range(24):
            cons = res["pt"][t] + res["pI"][t] - res["pE"][t]
            val = (cons - res["pI"][t]) / cons if cons > 0 else 0
            vals.append(val)
        plt.plot(vals, label=format_label(name))
    plt.title(f"{format_group_title(scenario_name)}: Self-Sufficiency Over Time")
    plt.xlabel("Hour")
    plt.ylabel("Self-Sufficiency [-]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Own-consumption over time = (consumption - import) / PV production
    plt.figure(figsize=(10, 5))
    base_own_cons = []
    for t in range(24):
        cons = base_results["pt"][t] + base_results["pI"][t] - base_results["pE"][t]
        pv = base_results["pt"][t]
        val = (cons - base_results["pI"][t]) / pv if pv > 0 else 0
        base_own_cons.append(val)
    plt.plot(base_own_cons, "--", color="black", label="Base Own Consumption")

    for name, res in scenario_results.items():
        vals = []
        for t in range(24):
            cons = res["pt"][t] + res["pI"][t] - res["pE"][t]
            pv = res["pt"][t]
            val = (cons - res["pI"][t]) / pv if pv > 0 else 0
            vals.append(val)
        plt.plot(vals, label=format_label(name))
    plt.title(f"{format_group_title(scenario_name)}: Own Consumption Over Time")
    plt.xlabel("Hour")
    plt.ylabel("Own Consumption [-]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Self_sufficiency and own consumption over a day for the different scenarios
    OC = []
    SS = []
    scenario_names = [format_label(name) for name in scenario_results.keys()]
    for name, res in scenario_results.items():
        cons = 0
        pv = 0
        pv_cons = 0
        for t in range(24):
            cons += res["pt"][t] + res["pI"][t] - res["pE"][t]
            pv += res["pt"][t]
            pv_cons += res["pt"][t] - res["pE"][t]
        ss = pv_cons/cons if cons>0 else 1
        oc = pv_cons/pv if pv>0 else 0
        OC.append(oc)
        SS.append(ss)
    
    x = np.arange(len(scenario_names))
    width = 0.35

    plt.bar(x - width/2, OC, width=width, color='orange', label='OC')
    plt.bar(x + width/2, SS, width=width, color='skyblue', label='SS')
    plt.title(f"{format_group_title(scenario_name)}: Own Consumption and Self-Sufficiency over Sensitivity scenarios")
    plt.xlabel("Scenario")
    plt.ylabel("Own Consumption [-]")
    plt.xticks(x, scenario_names) 
    plt.legend()
    plt.grid(True)
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


# PLOT for sensitivity analysis in question 1.c
def plot_1c_objfunc_sensitivity(scenario_results_grouped):
    for param, alpha_scenarios in scenario_results_grouped.items():
        
        alpha_values = []
        objectives_no_battery = []
        objectives_battery = []

        for alpha_val, scenarios in alpha_scenarios.items():
            alpha_values.append(alpha_val)
            objectives_no_battery.append(scenarios["No_Battery"]["objective_value"])
            objectives_battery.append(scenarios["Battery"]["objective_value"])

        x = np.arange(len(alpha_values))
        width = 0.35

        plt.figure(figsize=(10, 6))
        change_in_obj = [nb - b for nb, b in zip(objectives_no_battery, objectives_battery)]
        plt.bar(x, objectives_no_battery, width=width, color="skyblue", label="No Battery")
        plt.bar(x, objectives_battery, width=width, color="orange", label="Battery")

        plt.plot(x, change_in_obj, color="red", marker="o", linestyle=" ", label="Difference (No Battery - Battery)")

        plt.title(f"{format_group_title(param)}: Objective Sensitivity")
        plt.ylabel("Total Cost [DKK]")
        plt.xlabel("Alpha Values")
        plt.xticks(x, [f"α={val}" for val in alpha_values])
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()
    
def plot_1c_objfunc_sensitivity_demand(merged_scenario_results):
    alpha_list = list(merged_scenario_results.keys())
    demand_types = list(next(iter(merged_scenario_results.values())).keys())

    width = 0.2

    plt.figure(figsize=(12, 8))
    x = np.arange(len(alpha_list))

    offset = 0.3

    colors = {'demand_industrial': ['blue', 'lightblue'], 'demand_office': ['green', 'lightgreen'], 'demand_base':['red', 'salmon']}

    differences = {demand_type: [] for demand_type in demand_types}

    for i, demand_type in enumerate(demand_types):
        
        no_battery_obj = [merged_scenario_results[alpha][demand_type]["No_Battery"]["objective_value"] for alpha in alpha_list]
        battery_obj = [merged_scenario_results[alpha][demand_type]["Battery"]["objective_value"] for alpha in alpha_list]

        differences[demand_type] = [nb - b for nb, b in zip(no_battery_obj, battery_obj)]

        plt.bar(x + i*offset, no_battery_obj, width=width, color=colors[demand_type][0], edgecolor='black', label=f"No Battery - {demand_type.split('_')[-1]}")
        plt.bar(x + i*offset, battery_obj, width=width, color=colors[demand_type][1], edgecolor='black', label=f"Battery - {demand_type.split('_')[-1]}")

    for i, demand_type in enumerate(demand_types):
        plt.plot(x + i*offset, differences[demand_type], marker='o', color='black', linestyle=' ', label="Difference in the objective function" if i==0 else "")

    plt.title("Objective Function by Alpha and Demand Type")
    plt.xlabel("Alpha Values")
    plt.ylabel("Objective Value")
    plt.xticks(x + offset, alpha_list)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# We modify the calculation of oc and ss when taking into account battery
def plot_oc_ss(scenario_results, param_name):
    OC = []
    SS = []
    scenario_names = [format_label(name) for name in scenario_results.keys()]
    for name, res in scenario_results.items():
        cons = 0
        pv = 0
        pv_cons = 0
        imp = 0
        if param_name=='lambda_t':
            print(res)
        for t in range(24):
            cons += res["pt"][t] + res["pI"][t] - res["pE"][t] - res["p_ch"][t] + res["p_dis"][t]
            pv += res["pt"][t]
            imp += res["pI"][t]
            pv_cons += res["pt"][t] - res["pE"][t]
        ss = (cons-imp)/cons if cons>0 else 1
        oc = (cons-imp)/pv if pv>0 else 0
        OC.append(oc)
        SS.append(ss)
    
    x = np.arange(len(scenario_names))
    width = 0.35

    plt.bar(x - width/2, OC, width=width, color='orange', label='OC')
    plt.bar(x + width/2, SS, width=width, color='skyblue', label='SS')
    plt.title(f"{format_group_title(param_name)}: Own Consumption and Self-Sufficiency over Sensitivity scenarios")
    plt.xlabel("Scenario")
    plt.ylabel("Own Consumption [-]")
    plt.xticks(x, scenario_names) 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()