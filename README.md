# Aerospace propulsion project
BEMT project - Aircraft propeller design and analysis

This work is conducted as part of the course of _Aerospace propulsion_ and aims to design and analyze an aircraft propeller of a distributed electric propulsion system. This propeller is driven by an independent electric motor operating at constant rotational speed.

There are three main files to look at:
- ```main.py```: Main script to run in order to obtain the results for all four parts of the project. It sequentially performs the optimal propeller design (Part 1), the BEMT analysis at takeoff conditions (Part 2), the parametric study of collective pitch effects across a range of advance ratios (Part 3), and the determination of the optimal collective pitch for cruise conditions (Part 4). Running this file generates all figures and prints all key numerical results.
- ```functions.py```: Contains all helper functions called by `main.py`, including the iterative design loop and the BEMT solver.
- ```convregence.py```: Standalone script that performs a convergence analysis of the design procedure with respect to the number of blade sections N considered.
