from docplex.mp.model import Model
import numpy as np
from copy import deepcopy

seed = 42


class ClassicalPO:
    def __init__(self, mu, sigma, cfg, verbose=False):
        self.mu = mu
        self.sigma = sigma
        self.verbose = verbose
        self.cfg = deepcopy(cfg)

        self.n = self.mu.shape[0]
        self._max_return_value = None

    def rel_err(self, target, measured):
        return abs((target - measured) / target)

    def mvo_miqp(self):
        """Solves MVO with cardinality constraint (MIQP)"""
        mdl = Model(name="mvo_miqp")

        w = mdl.continuous_var_list(range(self.n), name="w", ub=1.0, lb=0.0)
        x = mdl.binary_var_list(range(self.n), name="x")

        l2_reg = self.cfg.gamma * mdl.sum_squares(w)
        objective = self.cfg.q * (w @ self.sigma @ w) - self.mu @ w + l2_reg

        # cardinality constraint
        mdl.add_constraint(mdl.sum(x) == self.cfg.kappa)

        mdl.add_constraints([a <= self.cfg.fmax * b for a, b in zip(w, x)])
        mdl.add_constraints([self.cfg.fmin * b <= a for a, b in zip(w, x)])

        # investment constraint
        mdl.add_constraint(mdl.sum(w) == 1.0)

        mdl.minimize(objective)
        mdl.context.solver.log_output = self.verbose
        mdl.parameters.randomseed = seed
        result = mdl.solve()

        return {
            "w": np.array(result.get_values(w)),
            "x": np.array(result.get_values(x)),
            "obj": result.objective_value,
        }

    def mvo_rc(self):
        """Solves MVO-RC (No Cardinality constraint)"""
        mdl = Model(name="mvo_rc")

        w = mdl.continuous_var_list(range(self.n), name="w", ub=1.0, lb=0.0)

        l2_reg = self.cfg.gamma * mdl.sum_squares(w)
        objective = self.cfg.q * (w @ self.sigma @ w) - self.mu @ w + l2_reg

        mdl.add_constraints([a <= self.cfg.fmax for a in w])
        mdl.add_constraints([self.cfg.fmin <= a for a in w])

        # investment constraint
        mdl.add_constraint(mdl.sum(w) == 1.0)

        mdl.minimize(objective)
        mdl.context.solver.log_output = self.verbose
        mdl.parameters.randomseed = seed
        result = mdl.solve()

        return {"w": np.array(result.get_values(w)), "obj": result.objective_value}

    def msr(self, k):
        """Solves MSR. Needs an upper bound `k` on variable `y` for imposing
        cardinaliy constraint"""

        mdl = Model(name="msr_rc")

        y = mdl.continuous_var_list(range(self.n), name="y", lb=0.0)
        x = mdl.binary_var_list(range(self.n), name="x")

        l2_reg = self.cfg.gamma * mdl.sum_squares(y)
        objective = (y @ self.sigma @ y) + l2_reg

        # cardinality constraint
        mdl.add_constraint(mdl.sum(x) == self.cfg.kappa)
        mdl.add_constraints([a <= self.cfg.fmax * k * b for a, b in zip(y, x)])
        mdl.add_constraints([self.cfg.fmin * k * b <= a for a, b in zip(y, x)])

        # investment constraint
        mdl.add_constraint(self.mu @ y == 1.0)

        mdl.minimize(objective)
        mdl.context.solver.log_output = self.verbose
        mdl.parameters.randomseed = seed
        result = mdl.solve()

        # inverse transform
        y_val = result.get_values(y)
        w_val = y_val / np.sum(y_val)

        return {
            "y": np.array(y_val),
            "w": np.array(w_val),
            "x": np.array(result.get_values(x)),
            "obj": result.objective_value,
        }

    def msr_rc(self):
        """Solves MSR-RC (No Cardinality constraint)"""
        mdl = Model(name="msr_rc")

        y = mdl.continuous_var_list(range(self.n), name="y", lb=0.0)
        k = mdl.continuous_var(name="k", lb=0.0)

        l2_reg = self.cfg.gamma * mdl.sum_squares(y)
        objective = (y @ self.sigma @ y) + l2_reg

        # investment constraint
        mdl.add_constraint(self.mu @ y == 1.0)

        mdl.add_constraint(mdl.sum(y) == k)
        mdl.add_constraints([a <= self.cfg.fmax * k for a in y])
        mdl.add_constraints([self.cfg.fmin * k <= a for a in y])

        mdl.minimize(objective)
        mdl.context.solver.log_output = self.verbose
        mdl.parameters.randomseed = seed
        result = mdl.solve()

        # inverse transform
        y_val = result.get_values(y)
        w_val = y_val / np.sum(y_val)

        return {
            "y": np.array(y_val),
            "w": np.array(w_val),
            "obj": result.objective_value,
        }

    def get_metrics(self, w):
        w = np.array(w).flatten()
        ret = self.mu @ w
        vol = np.sqrt(w @ self.sigma @ w)
        w_vol = np.sqrt(np.diag(self.sigma)) @ w

        return {
            "returns": ret,
            "risk": vol,
            "sharpe_ratio": ret / vol,
            "diversification_ratio": w_vol / vol,
        }
