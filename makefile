.PHONY: clean, all, FIG_PATHS

FIG_PATHS = fig/dynamics_examples.png
SIM_PATHS = throughput/Eigenvector_birfurcation.txt, throughput/Root_Degree_bifurcation.txt, SpringRank_bifurcation.txt	

figs: $(FIG_PATHS)
sims: $(SIM_PATHS)

all: figs

clean:
	rm -f $(FIGS)

$(FIG_PATHS): 
	python3 scripts/make_dynamics_fig.py

$(SIM_PATHS):
	python3 scripts/simulate_bifurcations.py




