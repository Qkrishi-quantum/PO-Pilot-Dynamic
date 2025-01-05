# import numpy as np
# import streamlit as st
# from dimod import ConstrainedQuadraticModel, Integer, quicksum
# from dwave.system import LeapHybridCQMSampler


# from itertools import product


# class SinglePeriod:
#     def __init__(
#         self,
#         alpha,
#         mu,
#         sigma,
#         budget,
#         price, # last day price for all the companies (array)
#         stock_names, # array
#         model_type="CQM",
#     ):
#         self.alpha = alpha
#         self.mu = mu
#         self.sigma = sigma
#         self.budget = budget
#         self.stock_prices = price
#         self.model_type = model_type
    
#         self.model = {"CQM": None}
#         self.sample_set = {}
    
#         self.sampler = {
#             "CQM": LeapHybridCQMSampler(token='DEV-e6d3f75883540313e43288ea60d6b62436fe0b46'),
#         }
    
#         self.solution = {}
#         self.precision = 2
    
#         self.max_num_shares = np.array((self.budget / self.stock_prices).astype(int))
#         self.stocks_names = stock_names
#         self.init_holdings = {s:0 for s in self.stocks_names}

#     def build_cqm(self, init_holdings):
#         # Instantiating the CQM object
#         cqm = ConstrainedQuadraticModel()

#         # Defining and adding variables to the CQM model
#         x = [
#             Integer(s, lower_bound=1, upper_bound=self.max_num_shares[i])
#             for i, s in enumerate(self.stocks_names)
#         ]

#         # Defining risk expression
#         risk = 0
#         stock_indices = range(len(self.stocks_names))
#         for s1, s2 in product(stock_indices, stock_indices):
#             coeff = (
#                 self.sigma[s1][s2]
#                 * self.stock_prices[s1]
#                 * self.stock_prices[s2]
#             )
#             risk = risk + coeff * x[s1] * x[s2]

#         # Defining the returns expression
#         returns = 0
#         for i in stock_indices:
#             returns = returns + self.stock_prices[i] * self.mu[i] * x[i]

#         if not init_holdings:
#             init_holdings = self.init_holdings
#         else:
#             self.init_holdings = init_holdings
        
#         cqm.add_constraint(
#             quicksum([x[i] * self.stock_prices[i] for i in stock_indices])
#             <= self.budget,
#             label="upper_budget",
#         )
#         cqm.add_constraint(
#             quicksum([x[i] * self.stock_prices[i] for i in stock_indices])
#             >= 0.997 * self.budget,
#             label="lower_budget",
#         )

#         # Objective: minimize mean-variance expression
#         cqm.set_objective(self.alpha * risk - returns)
#         cqm.substitute_self_loops()

#         self.model["CQM"] = cqm

#     def solve_cqm(self, init_holdings):
#         self.build_cqm(init_holdings)

#         # dwave-hardware
#         self.sample_set["CQM"] = self.sampler["CQM"].sample_cqm(
#             self.model["CQM"], label="CQM - Portfolio Optimization"
#         )

#         n_samples = len(self.sample_set["CQM"].record)
#         feasible_samples = self.sample_set["CQM"].filter(lambda d: d.is_feasible)

#         # feasible_samples = [1,2,3]
#         # n_samples = 32
        
#         if not feasible_samples:
#             raise Exception(  # pylint: disable=broad-exception-raised
#                 "No feasible solution could be found for this problem instance."
#             )
#         else:
#             solution = {}
            
#             best_feasible = feasible_samples.first
#             solution["stocks"] = {
#                 k: int(best_feasible.sample[k]) for k in self.stocks_names
#             }

#             # solution['stocks'] = {
#             #     k: np.random.randint(int(self.max_num_shares[idx]/2.0) + 1) for idx, k in enumerate(self.stocks_names)
#             # }

        
#             # spending = sum(
#             #     [
#             #         self.stock_prices[i]
#             #         * max(0, solution["stocks"][s] - self.init_holdings[s])
#             #         for i, s in enumerate(self.stocks_names)
#             #     ]
#             # )

#             # infosys
#             # initial_holding = 35
#             # solution[infosys] = 30
#             # 35-30 = 5
#             #
#             # sales = sum(
#             #     [
#             #         self.stock_prices[i]
#             #         * max(0, self.init_holdings[s] - solution["stocks"][s])
#             #         for i, s in enumerate(self.stocks_names)
#             #     ]
#             # )
            
#             # solution["investment_amount"] = spending + sales

#             # print(f"Sales Revenue: {sales:.2f}")
#             # print(f"Purchase Cost: {spending:.2f}")
#             # print(f"investment_amount Cost: {solution['investment_amount']:.2f}")

#             return solution
            
#     def _weight_allocation(self, stock_allocation_dict):
#         # Calculate the total value of the portfolio

#         portfolio_value = sum(
#             shares * price
#             for shares, price in zip(stock_allocation_dict.values(), self.stock_prices)
#         )

#         # Calculate individual asset weights
#         asset_weights = [
#             shares * price / portfolio_value
#             for shares, price in zip(stock_allocation_dict.values(), self.stock_prices)
#         ]

#         return asset_weights

#     def _get_optimal_weights_dict(self, asset_weights, stock_allocation_dict):
#         return dict(zip(stock_allocation_dict.keys(), np.round(asset_weights, 2)))

#     def _get_risk_ret(self, asset_weights):
#         # returns of a portfolio after optimum weight allocation

#         ret = np.sum(self.mu * asset_weights) 

#         # risk of a portfolio after optimum weight allocation
#         vol = np.sqrt(
#             np.dot(
#                 np.array(asset_weights).T,
#                 np.dot(self.sigma, asset_weights),
#             )
#         )

#         # sharpe ratio of a portfolio after optimum weight allocation_qu
#         sharpe_ratio = ret / vol

#         risk_ret_dict = {
#             "returns": np.round(ret * 100, 2),
#             "risk": np.round(vol * 100, 2),
#             "sharpe_ratio": np.round(sharpe_ratio, 2),
#         }

#         return risk_ret_dict






import numpy as np
import streamlit as st
from dimod import ConstrainedQuadraticModel, Integer, quicksum
from dwave.system import LeapHybridCQMSampler

import random


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

        self.tokens = {0:'DEV-17c9f3687f91e319a83e526e1dcdfbe53c3e0f7d',
                       1:'DEV-1e211edcb1e08a2568c1fa5676b7a1bae02ecbcd',
                       2:'DEV-ab89b7cb7a2c1720a03da9ef4f0f7bac61019db3',
                       3:'DEV-060b85a913971aee77fcd580ef0299071ccf732b',
                       4:'DEV-30343f6f7bbc8b88b069c11b877b91d6aaf29ebc',
                       5:'DEV-982350f82688aa63da31ce5d1e51f3131e6e1ccc',
                       6:'DEV-88084d4136ffa27511c371a44c1a8cbf4f94aaf3',
                       7:'DEV-bff296e5ca302de3718963ff2cb1532ad63de845',
                       8:'DEV-0fb34d846c85d1e57c7974c2679751c459594c1d',
                       9:'DEV-bff296e5ca302de3718963ff2cb1532ad63de845'}


    
        self.sampler = {
            "CQM": LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
            #0: LeapHybridCQMSampler(token='DEV-17c9f3687f91e319a83e526e1dcdfbe53c3e0f7d'),
            #1: LeapHybridCQMSampler(token='DEV-1e211edcb1e08a2568c1fa5676b7a1bae02ecbcd'),
            #2: LeapHybridCQMSampler(token='DEV-ab89b7cb7a2c1720a03da9ef4f0f7bac61019db3'),
            #3: LeapHybridCQMSampler(token='DEV-060b85a913971aee77fcd580ef0299071ccf732b'),
            #4: LeapHybridCQMSampler(token='DEV-30343f6f7bbc8b88b069c11b877b91d6aaf29ebc'),
            #5: LeapHybridCQMSampler(token='DEV-982350f82688aa63da31ce5d1e51f3131e6e1ccc'),
            6: LeapHybridCQMSampler(token='DEV-88084d4136ffa27511c371a44c1a8cbf4f94aaf3'),
            #7: LeapHybridCQMSampler(token='DEV-bff296e5ca302de3718963ff2cb1532ad63de845'),
            #8: LeapHybridCQMSampler(token='DEV-0fb34d846c85d1e57c7974c2679751c459594c1d'),
            #9: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336')
            10: LeapHybridCQMSampler(token='DEV-87a4d2b4988353fe5398fae9c73822f4939fe1a3'), # trialmailqkrishi@gmail.com
            11: LeapHybridCQMSampler(token='DEV-681fd08a3deaadeb703eca334c5f3859aea9b69d') # trialmailqkrishi2@gmail.com

        }


        # self.sampler = {
        #     "CQM": LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     0: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     1: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     2: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     3: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     4: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     5: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     6: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     7: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     8: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336'),
        #     9: LeapHybridCQMSampler(token='DEV-62e9d2528a9288aadf840016519e803a5b9e5336')
           
        # }


    
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

        return cqm     ########################################################## Added code

    def solve_cqm(self, init_holdings):
        cqm = self.build_cqm(init_holdings)     ############################################ Added code

        #random_token = random.randint(0,9)    ################################################   Added code

        
        feasible_samples = None

        for i in range(0,10):
            
            random_token = random.choice([6, 10, 11])
            #random_token = random.randint(0,9)
            print(f"token id:{random_token}")

            try:
                self.sample_set[random_token] = self.sampler[random_token].sample_cqm(
                    cqm, label=f"{random_token} id - portfolio optimization"
                )
                    
                #print("after_solving_cqm")
                
                feasible_samples = self.sample_set[random_token].filter(lambda d: d.is_feasible)
                  
            except Exception as e:
                print(f"The error :'{e}'' is being handled")
                print(f"{random_token} expired, Don't worry, trying again with other token")
                continue

            else:
                #print("in else")
                break
        
        if not feasible_samples:
            raise Exception(  # pylint: disable=broad-exception-raised
                "No feasible solution could be found for this problem instance.")
                
        else:
            solution = {}
            best_feasible = feasible_samples.first
            solution["stocks"] = {
            k: int(best_feasible.sample[k]) for k in self.stocks_names}

        # dwave-hardware
        # self.sample_set["CQM"] = self.sampler["CQM"].sample_cqm(
        #     self.model["CQM"], label="CQM - Portfolio Optimization"
        # )

        # n_samples = len(self.sample_set["CQM"].record)
        # feasible_samples = self.sample_set["CQM"].filter(lambda d: d.is_feasible)

        # feasible_samples = [1,2,3]
        # n_samples = 32
        
        # if not feasible_samples:
        #     raise Exception(  # pylint: disable=broad-exception-raised
        #         "No feasible solution could be found for this problem instance."
        #     )
        # else:
        #     solution = {}
            
        #     best_feasible = feasible_samples.first
        #     solution["stocks"] = {
        #         k: int(best_feasible.sample[k]) for k in self.stocks_names
        #     }

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
