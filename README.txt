
PART III PROJECT

This is a collection of scripts used for my Part III Chemistry project.

I am investigating the application of Gaussian Process Regression (GPR) to the Asakura Oosawa (AO) model for colloids in a polymer solution. While an analytical solution for the free energy of pairwise interactions exists, the three-body and higher terms are unknown. I am attempting to use GPR to learn this three-body term, and to use the model in molecular simulations. 

Essentially:

Create GPR model -> Verify GPR model -> use GPR model for physical calculations -> compare calculations to benchmarks


Script Descriptions:

AO_model_MC generates a system of colloids with random intercolloid distances and computes the free volume using a Monte Carlo approximation.

AO_data_gen calls AO_model_MC multiple times to generate a .csv file containing free volume and distance information, as well as MC error estimates. These models are contained in the model file.

AO_GP uses SKlearn's implementation of GPR to build 2 and 3 body models

AO_remainder computes the triple overlap volume and generates a GPR model for it

AO_learning generates learning curves to validate GPR models

AO_pairwise_ham contains the analytical AO pairwise solution

AO_metropolis uses the metropolis algorithm to find equlibrium configurations in the canonical ensemble. Can employ the 2-body, 3-body or analytical model

AO_metropolis_grid is an improved version of AO_metropolis, and employs a grid system to drastically speed up calculations

AO_rdf plots g(r) - the so-called "Radial Distribution Function" - for the output of AO_metropolis

AO_rdf_compare compares different rdf plots against each other

AO_full_grand performs a full simulation in the grand canonical ensemble - both colloids and polymers can be inserted/removed. Can be used to compute phase transitions

AO_half_grand treats colloids canonically, and polymers grand canonically. Can be used to generate benchmarks for our models





