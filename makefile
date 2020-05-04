.PHONY: clean, all

FIGS: fig/dynamics_examples.png
# , fig/bifurcations.png

SIMULATIONS: throughput/Eigenvector_birfurcation.txt, throughput/Root_Degree_bifurcation.txt, SpringRank_bifurcation.txt	

all: fig/dynamics_examples.png

fig/dynamics_examples.png: 
	python3 scripts/make_dynamics_fig.py

# fig/bifurcations.png: 


$(SIMULATIONS):
	python3 scripts/simulate_bifurcations.py


