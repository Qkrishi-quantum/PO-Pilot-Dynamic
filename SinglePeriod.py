import numpy as np
import streamlit as st
from dimod import ConstrainedQuadraticModel, Integer, quicksum
from dwave.system import LeapHybridCQMSampler


from itertools import product


class SinglePeriod:
    def __init__(
        self,
        alpha,
        mu,
        sigma,
        budget,
        price, # last day price for all the companies (array)
        stock_names, # array
        model_type="CQM",
    ):
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.budget = budget
        self.stock_prices = price
        self.model_type = model_type
    
        self.model = {"CQM": None}
        self.sample_set = {}
    
        self.sampler = {
            "CQM": LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        }
    
        self.solution = {}
        self.precision = 2
    
        self.max_num_shares = np.array((self.budget / self.stock_prices).astype(int))
        self.stocks_names = stock_names
        self.init_holdings = {s:0 for s in self.stocks_names}

    def build_cqm(self, init_holdings):
        # Instantiating the CQM object
        cqm = ConstrainedQuadraticModel()

        # Defining and adding variables to the CQM model
        x = [
            Integer(s, lower_bound=1, upper_bound=self.max_num_shares[i])
            for i, s in enumerate(self.stocks_names)
        ]

        # Defining risk expression
        risk = 0
        stock_indices = range(len(self.stocks_names))
        for s1, s2 in product(stock_indices, stock_indices):
            coeff = (
                self.sigma[s1][s2]
                * self.stock_prices[s1]
                * self.stock_prices[s2]
            )
            risk = risk + coeff * x[s1] * x[s2]

        # Defining the returns expression
        returns = 0
        for i in stock_indices:
            returns = returns + self.stock_prices[i] * self.mu[i] * x[i]

        if not init_holdings:
            init_holdings = self.init_holdings
        else:
            self.init_holdings = init_holdings
        
        cqm.add_constraint(
            quicksum([x[i] * self.stock_prices[i] for i in stock_indices])
            <= self.budget,
            label="upper_budget",
        )
        cqm.add_constraint(
            quicksum([x[i] * self.stock_prices[i] for i in stock_indices])
            >= 0.997 * self.budget,
            label="lower_budget",
        )

        # Objective: minimize mean-variance expression
        cqm.set_objective(self.alpha * risk - returns)
        cqm.substitute_self_loops()

        self.model["CQM"] = cqm

    def solve_cqm(self, init_holdings):
        self.build_cqm(init_holdings)

        # dwave-hardware
        self.sample_set["CQM"] = self.sampler["CQM"].sample_cqm(
            self.model["CQM"], label="CQM - Portfolio Optimization"
        )

        n_samples = len(self.sample_set["CQM"].record)
        feasible_samples = self.sample_set["CQM"].filter(lambda d: d.is_feasible)

        # feasible_samples = [1,2,3]
        # n_samples = 32
        
        if not feasible_samples:
            raise Exception(  # pylint: disable=broad-exception-raised
                "No feasible solution could be found for this problem instance."
            )
        else:
            solution = {}
            
            best_feasible = feasible_samples.first
            solution["stocks"] = {
                k: int(best_feasible.sample[k]) for k in self.stocks_names
            }

            # solution['stocks'] = {
            #     k: np.random.randint(int(self.max_num_shares[idx]/2.0) + 1) for idx, k in enumerate(self.stocks_names)
            # }

        
            # spending = sum(
            #     [
            #         self.stock_prices[i]
            #         * max(0, solution["stocks"][s] - self.init_holdings[s])
            #         for i, s in enumerate(self.stocks_names)
            #     ]
            # )

            # infosys
            # initial_holding = 35
            # solution[infosys] = 30
            # 35-30 = 5
            #
            # sales = sum(
            #     [
            #         self.stock_prices[i]
            #         * max(0, self.init_holdings[s] - solution["stocks"][s])
            #         for i, s in enumerate(self.stocks_names)
            #     ]
            # )
            
            # solution["investment_amount"] = spending + sales

            # print(f"Sales Revenue: {sales:.2f}")
            # print(f"Purchase Cost: {spending:.2f}")
            # print(f"investment_amount Cost: {solution['investment_amount']:.2f}")

            return solution
            
    def _weight_allocation(self, stock_allocation_dict):
        # Calculate the total value of the portfolio

        portfolio_value = sum(
            shares * price
            for shares, price in zip(stock_allocation_dict.values(), self.stock_prices)
        )

        # Calculate individual asset weights
        asset_weights = [
            shares * price / portfolio_value
            for shares, price in zip(stock_allocation_dict.values(), self.stock_prices)
        ]

        return asset_weights

    def _get_optimal_weights_dict(self, asset_weights, stock_allocation_dict):
        return dict(zip(stock_allocation_dict.keys(), np.round(asset_weights, 2)))

    def _get_risk_ret(self, asset_weights):
        # returns of a portfolio after optimum weight allocation

        ret = np.sum(self.mu * asset_weights) 

        # risk of a portfolio after optimum weight allocation
        vol = np.sqrt(
            np.dot(
                np.array(asset_weights).T,
                np.dot(self.sigma, asset_weights),
            )
        )

        # sharpe ratio of a portfolio after optimum weight allocation_qu
        sharpe_ratio = ret / vol

        risk_ret_dict = {
            "returns": np.round(ret * 100, 2),
            "risk": np.round(vol * 100, 2),
            "sharpe_ratio": np.round(sharpe_ratio, 2),
        }

        return risk_ret_dict