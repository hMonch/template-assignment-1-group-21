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
