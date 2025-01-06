# Conditional McKean-Vlasov Control With Deep Learning

This is a code that accompanies the paper titled [Deep Learning for Quadratic Hedging in Incomplete Jump Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4760864) that is a joint work with prof. Dr. Nacira Agram.
In particular, we develop an algorithm that learns the optimal control in jump-diffusion models with the inclusion of common noise, where the latter is modelled by regression on signatures of paths. 

## Code Structure

* **DataGenerator.py**: the artificial data in terms of Brownian and Poissonian increments is generated
* **DeepSolver.py**: Deep learning algorithm for learning the optimal control
* **MathModel.py**: Mathematical models for control problems used in computations
* **Net.py**: Neural network architecture
* **Tests.py**: The main file that should be run to obtain the results. Includes initializations of models as used in the paper.

## Requirements

Due to the requirements of the *signatory* package, this project uses Python 3.7. We refer you to their project [website](https://pypi.org/project/signatory/) for further information, especially regarding the installation.
