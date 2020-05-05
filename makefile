INDICATORS = throughput/equilibria_indicator.txt throughput/sim_indicator.txt

all: figs
figs: fig/dynamics_examples.png fig/bifurcations.png
indicators: $(INDICATORS)

clean_figs:
	rm -f fig/dynamics_examples.png
	rm -f fig/bifurcations.png

clean: 
	rm -f throughput/*.txt
	rm -f throughput/*.csv

throughput/sim_indicator.txt: scripts/simulate_bifurcations.py
	touch throughput/sim_indicator.txt
	python3 scripts/simulate_bifurcations.py
	
throughput/equilibria_indicator.txt: scripts/compute_equilibria.py
	touch throughput/equilibria_indicator.txt
	python3 scripts/compute_equilibria.py

fig/dynamics_examples.png: scripts/make_dynamics_fig.py
	python3 scripts/make_dynamics_fig.py

fig/bifurcations.png: scripts/make_bifurcation_fig.py indicators
	python3 scripts/make_bifurcation_fig.py