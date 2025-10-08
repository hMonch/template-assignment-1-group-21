from pathlib import Path
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class OptModel:
    """
    Optimization model for single consumer with flexible load and rooftop PV.
    """

    def __init__(self, data):
        """
        data: dictionary containing all input data
            - Pt: array of PV max production per hour [kW]
            - Dtot: total daily energy demand [kWh]
            - Dhour: max hourly load [kWh]
            - P_I: max grid import [kW]
            - P_E: max grid export [kW]
            - lambda_t: hourly price [DKK/kWh]
            - F_I: import tariff
            - F_E: export tariff
        """
        self.data = data
        self.results = {}
        self._build_model()

    def _build_variables(self):
        self.pt = [self.model.addVar(lb=0, ub=self.data["Pt"][t], name=f"pt_{t}") for t in range(24)]
        self.pI = [self.model.addVar(lb=0, ub=self.data["P_I"], name=f"pI_{t}") for t in range(24)]
        self.pE = [self.model.addVar(lb=0, ub=self.data["P_E"], name=f"pE_{t}") for t in range(24)]

    def _build_constraints(self):
        # Total energy demand constraint
        self.model.addConstr(
            gp.quicksum(self.pt[t] + self.pI[t] - self.pE[t] for t in range(24)) >= self.data["Dtot"],
            name="total_daily_demand"
        )

        # Hourly load limits
        for t in range(24):
            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] <= self.data["Dhour"],
                name=f"hourly_max_load_{t}"
            )
            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] >= 0,
                name=f"hourly_min_load_{t}"
            )

    def _build_objective_function(self):
        obj = gp.quicksum(
            self.pI[t] * (self.data["lambda_t"][t] + self.data["F_I"]) -
            self.pE[t] * (self.data["lambda_t"][t] - self.data["F_E"])
            for t in range(24)
        )
        self.model.setObjective(obj, GRB.MINIMIZE)

    def _build_model(self):
        self.model = gp.Model("FlexibleLoad_PV_Optimization")
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print("Optimization not successful")

    def _save_results(self):
        self.results["objective_value"] = self.model.ObjVal
        self.results["pt"] = [self.pt[t].x for t in range(24)]
        self.results["pI"] = [self.pI[t].x for t in range(24)]
        self.results["pE"] = [self.pE[t].x for t in range(24)]

    def display_results(self):
        print("\n------ Optimization Results ------")
        print(f"Optimal total cost: {self.results['objective_value']:.2f} DKK")
        print("Hourly PV production:", np.round(self.results["pt"], 2))
        print("Hourly grid import:", np.round(self.results["pI"], 2))
        print("Hourly grid export:", np.round(self.results["pE"], 2))


class OptModelFlex:

    def __init__(self, data, alpha):
        """
        data: dictionary containing all input data
            - Pt: array of hourly PV max production [kW]
            - Dhour: max hourly load [kWh]
            - lambda_t: hourly price [DKK/kWh]
            - F_I: import tariff
            - F_E: export tariff
            - dt: reference hourly load [kWh]
        alpha: penalty for shifted consumption (€/kWh or DKK/kWh)
        """
        self.data = data
        self.alpha = alpha
        self.results = {}
        self._build_model()

    def _build_variables(self):
        # Decision variables
        self.pt = [self.model.addVar(lb=0, ub=self.data["Pt"][t], name=f"pt_{t}") for t in range(24)]
        self.pI = [self.model.addVar(lb=0, ub=self.data["P_I"], name=f"pI_{t}") for t in range(24)]
        self.pE = [self.model.addVar(lb=0, ub=self.data["P_E"], name=f"pE_{t}") for t in range(24)]
        self.rho = [self.model.addVar(lb=0, name=f"rho_{t}") for t in range(24)]
        # Auxiliary variables for absolute value
        self.x_pos = [self.model.addVar(lb=0, name=f"x_pos_{t}") for t in range(24)]
        self.x_neg = [self.model.addVar(lb=0, name=f"x_neg_{t}") for t in range(24)]

    def _build_constraints(self):
        for t in range(24):
            # Hourly max/min load constraints
            
            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] >= 0,
                name=f"hourly_min_load_{t}"
            )

            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] <= self.data["Dhour"],
                name=f"hourly_max_load_{t}"
            )

            # Linearize absolute deviation for rho
            # x = actual load - reference load
            self.model.addConstr(
                (self.pt[t] + self.pI[t] - self.pE[t]) - self.data["dt"][t] == self.x_pos[t] - self.x_neg[t],
                name=f"x_decomp_{t}"
            )
            # Define rho as absolute fraction of deviation
            self.model.addConstr(
                self.rho[t] == (self.x_pos[t] + self.x_neg[t]) / self.data["dt"][t],
                name=f"rho_def_{t}"
            )

    def _build_objective_function(self):
        # Minimize cost + penalty for shifting
        obj = gp.quicksum(
            self.pI[t] * (self.data["lambda_t"][t] + self.data["F_I"]) -
            self.pE[t] * (self.data["lambda_t"][t] - self.data["F_E"]) +
            self.alpha * self.rho[t] * self.data["dt"][t]
            for t in range(24)
        )
        self.model.setObjective(obj, GRB.MINIMIZE)

    def _build_model(self):
        self.model = gp.Model("FlexibleLoad_PV_Optimization_Flex")
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print("Optimization not successful")

    def _save_results(self):
        self.results["objective_value"] = self.model.ObjVal
        self.results["pt"] = [self.pt[t].x for t in range(24)]
        self.results["pI"] = [self.pI[t].x for t in range(24)]
        self.results["pE"] = [self.pE[t].x for t in range(24)]
        self.results["rho"] = [self.rho[t].x for t in range(24)]

    def display_results(self):
        print("\n------ Optimization Results (Flex) ------")
        print(f"Optimal total cost: {self.results['objective_value']:.2f} DKK")
        print("Hourly PV production:", np.round(self.results["pt"], 2))
        print("Hourly grid import:", np.round(self.results["pI"], 2))
        print("Hourly grid export:", np.round(self.results["pE"], 2))
        print("Hourly shift fraction rho:", np.round(self.results["rho"], 2))

class OptModelFlexBattery:

    def __init__(self, data, alpha):
        """
        data: dictionary containing all input data including battery params
            - Pt: hourly PV production [kW]
            - Dhour: max hourly load [kWh]
            - lambda_t: hourly price [DKK/kWh]
            - F_I, F_E: import/export tariffs
            - dt: reference hourly load [kWh]
            - C: battery capacity [kWh]
            - eta_ch, eta_dis: efficiencies
            - P_ch, P_dis: charge/discharge limits [kW]
            - initial_soc_ratio, final_soc_ratio
        alpha: penalty for shifted consumption (€/kWh or DKK/kWh)
        """
        self.data = data
        self.alpha = alpha
        self.results = {}
        self._build_model()

    # -------------------------------------------------------------------------
    def _build_variables(self):
        self.pt = [self.model.addVar(lb=0, ub=self.data["Pt"][t], name=f"pt_{t}") for t in range(24)]
        self.pI = [self.model.addVar(lb=0, ub=self.data["P_I"], name=f"pI_{t}") for t in range(24)]
        self.pE = [self.model.addVar(lb=0, ub=self.data["P_E"], name=f"pE_{t}") for t in range(24)]
        self.p_ch = [self.model.addVar(lb=0, ub=self.data["P_ch"], name=f"p_ch_{t}") for t in range(24)]
        self.p_dis = [self.model.addVar(lb=0, ub=self.data["P_dis"], name=f"p_dis_{t}") for t in range(24)]
        self.E = [self.model.addVar(lb=0, ub=self.data["C"], name=f"E_{t}") for t in range(25)]  # 0–24 SOC
        self.rho = [self.model.addVar(lb=0, name=f"rho_{t}") for t in range(24)]
        self.x_pos = [self.model.addVar(lb=0, name=f"x_pos_{t}") for t in range(24)]
        self.x_neg = [self.model.addVar(lb=0, name=f"x_neg_{t}") for t in range(24)]

    # -------------------------------------------------------------------------
    def _build_constraints(self):
        C = self.data["C"]
        eta_ch = self.data["eta_ch"]
        eta_dis = self.data["eta_dis"]

        # --- Battery initial and final SOC ---
        self.model.addConstr(self.E[0] == C * self.data["initial_soc_ratio"], name="E_init")
        self.model.addConstr(self.E[24] == C * self.data["final_soc_ratio"], name="E_final")

        # --- Battery dynamics ---
        for t in range(24):
            # Energy balance
            self.model.addConstr(
                self.E[t + 1] == self.E[t] + eta_ch * self.p_ch[t] - (1 / eta_dis) * self.p_dis[t],
                name=f"soc_balance_{t}"
            )

            # Hourly power flow constraint with battery considered
            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] + self.p_dis[t] - self.p_ch[t] >= 0,
                name=f"hourly_min_load_{t}"
            )

            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] + self.p_dis[t] - self.p_ch[t] <= self.data["Dhour"],
                name=f"hourly_max_load_{t}"
            )

            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] + self.p_dis[t] - self.p_ch[t] >= 0,
                name=f"positive_consumption_{t}"
            )

            # --- Linearize absolute deviation for rho (modified to include battery) ---
            self.model.addConstr(
                (self.pt[t] + self.pI[t] - self.pE[t] + self.p_dis[t] - self.p_ch[t]) - self.data["dt"][t]
                == self.x_pos[t] - self.x_neg[t],
                name=f"x_decomp_{t}"
            )

            self.model.addConstr(
                self.rho[t] == (self.x_pos[t] + self.x_neg[t]) / self.data["dt"][t],
                name=f"rho_def_{t}"
            )

    # -------------------------------------------------------------------------
    def _build_objective_function(self):
        obj = gp.quicksum(
            self.pI[t] * (self.data["lambda_t"][t] + self.data["F_I"]) -
            self.pE[t] * (self.data["lambda_t"][t] - self.data["F_E"]) +
            self.alpha * self.rho[t] * self.data["dt"][t]
            for t in range(24)
        )
        self.model.setObjective(obj, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    def _build_model(self):
        self.model = gp.Model("FlexibleLoad_PV_Optimization_Flex_Battery")
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    # -------------------------------------------------------------------------
    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print("Optimization not successful")

    # -------------------------------------------------------------------------
    def _save_results(self):
        self.results["objective_value"] = self.model.ObjVal
        self.results["pt"] = [self.pt[t].x for t in range(24)]
        self.results["pI"] = [self.pI[t].x for t in range(24)]
        self.results["pE"] = [self.pE[t].x for t in range(24)]
        self.results["p_ch"] = [self.p_ch[t].x for t in range(24)]
        self.results["p_dis"] = [self.p_dis[t].x for t in range(24)]
        self.results["E"] = [self.E[t].x for t in range(25)]
        self.results["rho"] = [self.rho[t].x for t in range(24)]

    # -------------------------------------------------------------------------
    def display_results(self):
        print("\n------ Optimization Results (Flex + Battery) ------")
        print(f"Optimal total cost: {self.results['objective_value']:.2f} DKK")
        print("Hourly PV production:", np.round(self.results["pt"], 2))
        print("Hourly grid import:", np.round(self.results["pI"], 2))
        print("Hourly grid export:", np.round(self.results["pE"], 2))
        print("Hourly battery charge:", np.round(self.results["p_ch"], 2))
        print("Hourly battery discharge:", np.round(self.results["p_dis"], 2))
        print("State of Charge (SOC):", np.round(self.results["E"], 2))
        print("Hourly shift fraction rho:", np.round(self.results["rho"], 2))

class OptModelFlexBatteryInvestment:
    def __init__(self, data, alpha, phi):
        """
        data: dictionary containing all input data including battery params
            - Pt: hourly PV production [kW]
            - Dhour: max hourly load [kWh]
            - lambda_t: hourly price [DKK/kWh]
            - F_I, F_E: import/export tariffs
            - dt: reference hourly load [kWh]
            - C: base battery capacity [kWh]
            - eta_ch, eta_dis: efficiencies
            - P_ch, P_dis: charge/discharge limits [kW]
            - initial_soc_ratio, final_soc_ratio
        alpha: penalty for shifted consumption (DKK/kWh)
        phi: capital cost per kWh of installed battery [DKK/kWh]
        """
        self.data = data
        self.alpha = alpha
        self.phi = phi  # cost coefficient for investment
        self.results = {}
        self._build_model()

    # -------------------------------------------------------------------------
    def _build_variables(self):
        # Hourly operational variables
        self.pt = [self.model.addVar(lb=0, ub=self.data["Pt"][t], name=f"pt_{t}") for t in range(24)]
        self.pI = [self.model.addVar(lb=0, ub=self.data["P_I"], name=f"pI_{t}") for t in range(24)]
        self.pE = [self.model.addVar(lb=0, ub=self.data["P_E"], name=f"pE_{t}") for t in range(24)]
        self.p_ch = [self.model.addVar(lb=0, name=f"p_ch_{t}") for t in range(24)]
        self.p_dis = [self.model.addVar(lb=0, name=f"p_dis_{t}") for t in range(24)]
        self.E = [self.model.addVar(lb=0, name=f"E_{t}") for t in range(25)]
        self.rho = [self.model.addVar(lb=0, name=f"rho_{t}") for t in range(24)]
        self.x_pos = [self.model.addVar(lb=0, name=f"x_pos_{t}") for t in range(24)]
        self.x_neg = [self.model.addVar(lb=0, name=f"x_neg_{t}") for t in range(24)]
        self.k = self.model.addVar(lb=0, ub=100, name="k")


    # -------------------------------------------------------------------------
    def _build_constraints(self):
        C = self.data["C"]
        eta_ch = self.data["eta_ch"]
        eta_dis = self.data["eta_dis"]

        # Battery initial/final state
        self.model.addConstr(self.E[0] == self.k * C * self.data["initial_soc_ratio"], name="E_init")
        self.model.addConstr(self.E[24] == self.k * C * self.data["final_soc_ratio"], name="E_final")

        for t in range(24):
            # SOC evolution
            self.model.addConstr(
                self.E[t + 1] == self.E[t] + eta_ch * self.p_ch[t] - (1 / eta_dis) * self.p_dis[t],
                name=f"soc_balance_{t}"
            )

            # Power and load balance
            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] + self.p_dis[t] - self.p_ch[t] >= 0,
                name=f"hourly_min_load_{t}"
            )
            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] + self.p_dis[t] - self.p_ch[t] <= self.data["Dhour"],
                name=f"hourly_max_load_{t}"
            )

            self.model.addConstr(
                self.pt[t] + self.pI[t] - self.pE[t] + self.p_dis[t] - self.p_ch[t] >= 0,
                name=f"positive_consumption_{t}"
            )

            # Battery power limits scaled by k
            self.model.addConstr(self.p_ch[t] * eta_ch <= self.k * self.data["P_ch"], name=f"charge_limit_{t}")
            self.model.addConstr(self.p_dis[t] / eta_dis <= self.k * self.data["P_dis"], name=f"discharge_limit_{t}")
            self.model.addConstr(self.E[t] <= self.k * C, name=f"soc_capacity_{t}")

            # Linearize absolute deviation (same as before)
            self.model.addConstr(
                (self.pt[t] + self.pI[t] - self.pE[t] + self.p_dis[t] - self.p_ch[t]) - self.data["dt"][t]
                == self.x_pos[t] - self.x_neg[t],
                name=f"x_decomp_{t}"
            )
            self.model.addConstr(
                self.rho[t] == (self.x_pos[t] + self.x_neg[t]) / self.data["dt"][t],
                name=f"rho_def_{t}"
            )

    # -------------------------------------------------------------------------
    def _build_objective_function(self):
        # Daily cost (operational + investment amortized)
        operational_cost = gp.quicksum(
            self.pI[t] * (self.data["lambda_t"][t] + self.data["F_I"]) -
            self.pE[t] * (self.data["lambda_t"][t] - self.data["F_E"]) +
            self.alpha * self.rho[t] * self.data["dt"][t]
            for t in range(24)
        )

        # Battery investment cost (converted to daily equivalent)
        daily_investment_cost = self.phi * self.k * self.data["C"] / (10 * 365)

        self.model.setObjective(operational_cost + daily_investment_cost, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    def _build_model(self):
        self.model = gp.Model("Flex_Battery_Investment")
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    # -------------------------------------------------------------------------
    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print("Optimization not successful")

    # -------------------------------------------------------------------------
    def _save_results(self):
        self.results["objective_value"] = self.model.ObjVal
        self.results["k"] = self.k.x
        self.results["pt"] = [self.pt[t].x for t in range(24)]
        self.results["pI"] = [self.pI[t].x for t in range(24)]
        self.results["pE"] = [self.pE[t].x for t in range(24)]
        self.results["p_ch"] = [self.p_ch[t].x for t in range(24)]
        self.results["p_dis"] = [self.p_dis[t].x for t in range(24)]
        self.results["E"] = [self.E[t].x for t in range(25)]
        self.results["rho"] = [self.rho[t].x for t in range(24)]

    # -------------------------------------------------------------------------
    def display_results(self):
        print("\n------ Optimization Results (Battery Investment) ------")
        print(f"Optimal total daily cost: {self.results['objective_value']:.2f} DKK")
        print(f"Optimal battery size factor k: {self.results['k']:.3f}")
        print("Hourly PV production:", np.round(self.results["pt"], 2))
        print("Hourly grid import:", np.round(self.results["pI"], 2))
        print("Hourly grid export:", np.round(self.results["pE"], 2))
        print("Hourly battery charge:", np.round(self.results["p_ch"], 2))
        print("Hourly battery discharge:", np.round(self.results["p_dis"], 2))
        print("State of Charge (SOC):", np.round(self.results["E"], 2))
        print("Hourly shift fraction rho:", np.round(self.results["rho"], 2))