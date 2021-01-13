Overview
========
This repository allows to compare the performance of the selection algorithms LEGACY and LEXIMIN, and to reproduce the
experiments in the following manuscript:
> Bailey Flanigan, Paul Gölz, Anupam Gupta, Brett Hennig, Ariel D. Procaccia. Fair Algorithms for Selecting Citizens'
> Assemblies. (2021)

While this repository contains the implementation of two selection algorithms, it is intended for scientific analysis
rather than for usage in actual citizens’ assemblies. In particular, this repository includes the versions of the
algorithms at the time of writing of this paper, which is why future improvements to these algorithms will not be
incorporated into this repository. For use in citizens’ assemblies, we recommend using the
[Stratification App](https://github.com/sortitionfoundation/stratification-app/) maintained by the Sortition Foundation.

This directory contains two scripts:
- [stratification.py](stratification.py) contains implementations of both selection algorithms. This file is taken from
  the stratification-app developed by the Sortition Foundation (methods and dependencies not necessary for the
  experiments have been removed).
- [analysis.py](analysis.py) allows to analyze a sampling instance (consisting of a file specifying features and quotas
  and a file specifying the members of a pool). The script runs LEGACY and LEXIMIN on the instance, compares the
  probability allocations resulting from these runs, charts which pool members are least frequently selected by LEGACY,
  and measures the running time of LEXIMIN.

System Requirements
===================
The script is written for Python 3.7 or higher, with the ILP solver Gurobi installed (commercial, but free academic
licenses are available). Additionally, recent versions of the following python packages must be installed:
- Python bindings for Gurobi
- Matplotlib
- Numpy
- Pandas
- Python-MIP
- Seaborn
- Scipy

We tested our software on the two following configurations:
- macOS 10.15.7 and Python 3.7.9, Gurobi 9.0.3, Matplotlib 3.0.2, Numpy 1.17.3, Pandas 1.0.3, Python-MIP 1.13.0,
  Seaborn 0.11.0, SciPy 1.2.0 (development platform, results and paper and reference outputs produced with this
  configuration), and
- Ubuntu Server 20.04.01 and Python 3.8.5, Gurobi 9.1.1, Matplotlib 3.3.3, Numpy 1.19.5, Pandas 1.2.0,
  Python-MIP 1.13.0, Seaborn 0.11.1, SciPy 1.6.0 (tested installation).

The code runs on standard hardware.

Installation guide
==================
Besides the above-mentioned dependencies, no installation is required. In the following, we give instructions for
installing the dependencies. These instructions assume that python 3.7 or higher are already installed as `python3`.
Familiarity with the command line is assumed.

The main difficulty of installation is obtaining a Gurobi license and going through the steps of installing Gurobi.
All parts of the installation themselves should run in seconds on a normal computer.

### Obtain a free academic license for Gurobi
- in a browser, visit https://www.gurobi.com/downloads/end-user-license-agreement-academic/
- log in or register
- potentially re-visit https://www.gurobi.com/downloads/end-user-license-agreement-academic/ , accept  
- at the bottom of the page, there is a code like `grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`, copy this code

### Install the Gurobi optimizer package
- follow the instructions of the software installation guide for
[Linux](https://www.gurobi.com/documentation/9.1/quickstart_linux/software_installation_guid.html),
[macOS](https://www.gurobi.com/documentation/9.1/quickstart_mac/software_installation_guid.html), or
[Windows](https://www.gurobi.com/documentation/9.1/quickstart_windows/software_installation_guid.html)
- activate the license by running the `grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` command copied before
- navigate into the package where Gurobi is installed (e.g., into `/opt/gurobi911/linux64`) and run
  `$ python3 setup.py install` to establish the gurobi bindings

### Install other Python dependencies
- if `pip3` is not yet installed, install it (e.g., by running `sudo apt install python3-pip` on Ubuntu)
- install all other packages (`pip3 install matplotlib numpy pandas mip seaborn scipy`)

To test that the installation is working properly, you might try to run the first command from the demo section.

Usage
=====
Input
-----
The script `analysis.py` can analyze any given panel-selection instance. The information about each instance must
be placed into a separate subdirectory of `./data/`. The name of this directory must be composed of two parts separated
by an underscore: The first part is the name of the instance, which can freely be chosen. The second part is the
desired panel size for the instance. For example, this directory might be called `./data/cca_75/` or `./data/sf_a_35/`.

Each instance directory must contain two files, `categories.csv` and `respondents.csv`, as follows:

### `categories.csv` file
`categories.csv` lists all features with their lower and upper quotas. Features are assumed to be organized in
categories (e.g., gender), where each feature is given by a specific value for this feature (e.g., female, male,
nonbinary). Each row contains a category and a corresponding feature, as well as the lower and upper quota. For example,
the table might start as follows:

| category | feature | min | max |
|----------|---------|-----|-----|
| gender   | female  | 49  | 51  |
| gender   | male    | 49  | 51  |
| gender   | other   | 1   | 1   |
| age      | 18-25   | 10  | 15  |
| age      | 25-50   | 14  | 18  |
| …        | …       | …   | …   |

### `respondents.csv` file
`respondents.csv` lists all pool members. Each row of the table corresponds to a single pool member and contains her
value for each of the feature categories. For example, this table might look as follows:

| gender | age     | … |
|--------|---------|---|
| female | 18–25   | … |
| other  | 70+     | … |
| female | 25–50   | … |
| female | 18–25   | … |
| …      | …       | … |

Calling the Script
------------------
To analyze an instance with name `instance_name` and panel size `panel_size`, run the script as `$ python3 analysis.py
instance_name panel_size`. For example, running `$ python3 analysis.py sf_a 35` would analyze the instance specified by
a directory `./data/sf_a_35` if present. The first time that this is done for an instance can take a fair amount of
time, especially since two sets of 10,000 panels are drawn via the algorithm LEGACY. Once these algorithms have run
once, their results are saved to disk (in the folder `./distributions/`) and subsequent calls use these saved results.

Without additional arguments, LEXIMIN is run three times at the end to measure its median running time, and these runs
are done every time the script is called. If the flag `--skiptiming` is specified, these runs are skipped instead, which
might reduce running time. 

More information on usage, as well as a list of instances in the data folder that could be analyzed, can be obtained by
running `python3 analysis.py --help`. 

Output
------
Calling the script produces all instance statistics mentioned in the paper:
- pool size n
- panel size k
- \# quota categories
- mean selection probability k/n
- LEGACY minimum probability (information for computing the confidence bound)
- LEXIMIN minimum probability
- gini coefficient of LEGACY
- gini coefficient of LEXIMIN
- geometric mean of LEGACY
- geometric mean of LEXIMIN
- share selected by LEGACY with probability below LEXIMIN minimum selection probability
- runtime of LEXIMIN algorithm (except if `--skiptiming` is set)

Together with output on the running of LEGACY and LEXIMIN, this information is written out to the console and is
additionally saved to a file `./analysis/[instance_name]_[panel_size]_statistics.txt`.

Additionally, the call to `analysis.py` creates two figures and there corresponding raw data:
- `./analysis/[instance_name]_[panel_size]_prob_allocs.pdf` is the equivalent to Figure D2 (of which Figures 2 and D1
  show subranges), visualizing the probability allocations of LEGACY and LEXIMIN. The raw data for this plot is saved to
  `./analysis/[instance_name]_[panel_size]_prob_allocs_data.csv`.
- `./analysis/[instance_name]_[panel_size]_ratio_product.pdf` is the equivalent to Figure D3, which plots whether pool
  members with more overrepresented features are more rarely selected. The raw data for this plot is saved to
  `./analysis/[instance_name]_[panel_size]_ratio_product_data.csv`.

Demo
====
While the panel data used in the paper is not publicly available, this repository contains two sample instances:
- instance `example_large` with panel size 200 (the example given in methods section M8) and
- instance `example_small` with panel size 20 (a smaller version, the instance Alternate(200, 20) according to the
  definitions in Section S6 of the supplementary information)
  
We will first run the analysis on the small example instance, which takes about 90 seconds:
```     
$ python3 analysis.py example_small 20
                                                                                                            pgoelz@GS17998
Running iteration 100 out of 10000.
Running iteration 200 out of 10000.
Running iteration 300 out of 10000.
… (progress of first LEGACY set)
Running iteration 10000 out of 10000.
Running iteration 100 out of 10000.
Running iteration 200 out of 10000.
… (progress of second LEGACY set)
Running iteration 10000 out of 10000.
Using license file /Users/username/gurobi.lic
Academic license - for non-commercial use only
Academic license - for non-commercial use only
Multiplicative weights phase, round 1/600. Discovered 1 committees so far.
Multiplicative weights phase, round 2/600. Discovered 2 committees so far.
Multiplicative weights phase, round 3/600. Discovered 3 committees so far.
… (progress of LEXIMIN run)
Multiplicative weights phase, round 600/600. Discovered 68 committees so far.
All agents are contained in some feasible committee.
Fixed 0/200 probabilities.
Maximin is at most 32.78%, can do 10.00% with 68 committees. Gap 22.78%.
Maximin is at most 27.52%, can do 10.00% with 69 committees. Gap 17.52%.
Maximin is at most 24.64%, can do 10.00% with 70 committees. Gap 14.64%.
… (progress of LEXIMIN run)
Maximin is at most 12.51%, can do 10.00% with 202 committees. Gap 2.51%.
Maximin is at most 13.31%, can do 10.00% with 203 committees. Gap 3.31%.
Maximin is at most 10.02%, can do 10.00% with 204 committees. Gap 0.02%.
instance: example_small
pool size n: 200
panel size k: 20
# quota categories: 2
mean selection probability k/n: 10.0%
LEGACY minimum probability: ≤ 1.21% (99% upper confidence bound based on Jeffreys interval for a binomial parameter, \
calculated from sample proportion 0.0096 and sample size 10,000)
LEXIMIN minimum probability (exact): 10.0%
gini coefficient of LEGACY: 2.1%
gini coefficient of LEXIMIN: 0.0%
geometric mean of LEGACY: 9.9%
geometric mean of LEXIMIN: 10.0%
share selected by LEGACY with probability below LEXIMIN minimum selection probability: 48.0%
Plot of probability allocation created at analysis/example_small_20_prob_allocs.pdf.
Plot of ratio products created at analysis/example_small_20_ratio_product.pdf.… (three more runs of LEXIMIN)
Run 3/3 of LEXIMIN took 6.4 seconds.
Out of 3 runs, LEXIMIN took a median running time of 6.4 seconds.
```
This should produce five files in `./analysis/`: `example_small_20_statistics.txt`,
`example_small_20_ratio_product.pdf`, `example_small_20_ratio_product_data.csv`, `example_small_20_prob_allocs.pdf`, and
`example_small_20_prob_allocs_data.csv`. Our version of these files can be found in the directory `reference_output`.

To try a challenging input for LEXIMIN, we also run the large example instance. This time, we skip the timing step (the
analysis will still take around an hour to run for the first time):
```
$ python3 analysis.py --skiptiming example_large 200

… (debug output of LEGACY and LEXIMIN)
Maximin is at most 10.14%, can do 10.00% with 2049 committees. Gap 0.14%.
Maximin is at most 10.00%, can do 10.00% with 2050 committees. Gap -0.00%.
instance: example_large
pool size n: 2000
panel size k: 200
# quota categories: 2
mean selection probability k/n: 10.0%
LEGACY minimum probability: ≤ 0.25% (99% upper confidence bound based on Jeffreys interval for a binomial parameter, \
calculated from sample proportion 0.0014 and sample size 10,000)
LEXIMIN minimum probability (exact): 10.0%
gini coefficient of LEGACY: 1.8%
gini coefficient of LEXIMIN: 0.0%
geometric mean of LEGACY: 10.0%
geometric mean of LEXIMIN: 10.0%
share selected by LEGACY with probability below LEXIMIN minimum selection probability: 49.9%
Plot of probability allocation created at analysis/example_large_200_prob_allocs.pdf.
Plot of ratio products created at analysis/example_large_200_ratio_product.pdf.
Skip timing.
```

Reproduction
============
To reproduce our results, place the directories with the real panel data into `./data/`, and run the following commands:
```
$ python3 analysis.py cca 75
$ python3 analysis.py hd 30
$ python3 analysis.py mass 24
$ python3 analysis.py nexus 170
$ python3 analysis.py obf 30
$ python3 analysis.py sf_a 35
$ python3 analysis.py sf_b 20
$ python3 analysis.py sf_c 44
$ python3 analysis.py sf_d 40
$ python3 analysis.py sf_e 110
```
As described in the section "Output" above, this will generate files containing the instance statistics mentioned in the
paper as well as versions of the figures. Reference output of all files produced is placed into `./reference_output/`.
