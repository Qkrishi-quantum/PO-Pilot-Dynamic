# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import dimod
from docplex.mp.model import Model
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp


def build_bqm(alpha, _mu, _sigma, cardinality):
           n = len(_mu)
           mdl = Model(name="stock_selection")
           x = mdl.binary_var_list(range(n), name="x")

           objective = alpha * (x @ _sigma @ x) - _mu @ x

           # cardinality constraint
           mdl.add_constraint(mdl.sum(x) == cardinality)
           mdl.minimize(objective)

           qp = from_docplex_mp(mdl)
           qubo = QuadraticProgramToQubo().convert(qp)

           bqm = dimod.as_bqm(
               qubo.objective.linear.to_array(),
               qubo.objective.quadratic.to_array(),
               dimod.BINARY,)
           return bqm