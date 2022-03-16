# Heavy Neutral Lepton Decay in Flight (HNL-DIF)

This code derives event rates of heavy neutrino decays in flight at a few accelerator neutrino experiments, and draws limits in the model parameter space.

All plots generated in the main text of [the associated publication](https://inspirehep.net/literature/1919114) can be generated with the *main_plots* jupyter notebook. 

The module *hnl_apps* takes care of computing HNL fluxes, decay rates, and load experimental and model parameters. 

It also contains a Monte Carlo tool to generate heavy neutrino events. This functionality is explored in the notebook called *plot_kin_distributions*.

---

### Notebooks:

* compute_constraints 

The event rate across relevent 2D parameter space is computed.

* main_plots

Plots all parameter space plots with constraints and draws the contour of event rates for our simulations

* plot_kin_distributions

Generates HNL decay events and plots several interesting kinematical variables. 

* side_checks

Computes low-level quantities like branching ratios, lifetimes, fluxes, etc.

* T2K_data_release_limits

Loads the data from the T2K data release and plots the resulting efficiencies, HNL fluxes, and corresponding limits.

--- 

If you use this code, please consider citing:

 ```@article{Arguelles:2021dqn,
    author = {Arg\"uelles, Carlos A. and Foppiani, Nicol\`o and Hostert, Matheus},
    title = "{Heavy neutral leptons below the kaon mass at hodoscopic detectors}",
    eprint = "2109.03831",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "9",
    year = "2021"}
```
