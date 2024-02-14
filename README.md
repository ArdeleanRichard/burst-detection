# burst-detection
Python implementation of neuronal burst detection algorithms and comparison between them on synthetic data.

Burst detection algorithms:
- ISIn
- ISI Rank Threshold (IRT)
- Cumulative Moving Average (CMA)
- Max Interval (MI)
- Rank Surprise (RS)
- Poisson Surprise (PS)

These algorithms have been compared on synthetic data through boxplots based on the percentage of true positives and false positives. 
Any of these algorithms can also be executed on real data provided in this repository.

Synthetic data:
```
A comparison of computational methods for detecting bursts in neuronal spike trains and their application to human stem cell-derived neuronal networks
https://github.com/ellesec/burstanalysis/tree/master/Simulation_results
```