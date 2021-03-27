# coding: utf-8
# Copyright (C) 2021 Paul Gölz and Bailey Flanigan

import csv
import operator
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from math import ceil
from pathlib import Path
from pickle import load, dump
from random import seed
from time import time
from typing import Dict, List, Any, NewType, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy.stats import beta
from scipy.stats.mstats import gmean
from stratification import find_random_sample_legacy, SelectionError, check_min_cats, find_distribution_leximin

AgentId = NewType("AgentId", Any)  # type of agent identifier
FeatureCategory = NewType("FeatureCategory", str)  # type for category of features such as "gender"
Feature = NewType("Feature", str)  # type for features within a category such as "female"
FeatureInfo = NewType("FeatureInfo", Dict[str, int])  # information about a feature, including quotas "min" and "max"
ProbAllocation = NewType("ProbAllocation", Dict[AgentId, float])  # type for probability allocations


@dataclass
class Instance:
    k: int
    categories: Dict[FeatureCategory, Dict[Feature, FeatureInfo]]
    agents: Dict[AgentId, Dict[FeatureCategory, Feature]]


def read_instance(feature_file: Union[str, Path], pool_file: Union[str, Path], k: int) -> Instance:
    """Read in an instance from two files, one specifying the features and the second one specifying the pool members.
    For format, see `README.md` and the example directories in `./data`."""
    feature_info: Dict[FeatureCategory, Dict[Feature, FeatureInfo]]  # information about all features, by category
    feature_info = {}

    with open(feature_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            category = FeatureCategory(line["category"])
            feature = Feature(line["feature"])
            mi = line["min"]
            ma = line["max"]
            if category not in feature_info:
                feature_info[category] = {}
            feature_info[category][feature] = FeatureInfo({"min": int(mi), "max": int(ma), "selected": 0,
                                                           "remaining": 0})

    categories = list(feature_info)
    agents: Dict[AgentId, Dict[FeatureCategory, Feature]]
    agents = {}
    with open(pool_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for i, line in enumerate(reader):
            agent_id = AgentId(i)
            agents[agent_id] = {category: Feature(line[category]) for category in categories}
            # LEGACY requires "remaining" to be initialized with the number of pool members with this feature
            for category in categories:
                feature_info[category][Feature(line[category])]["remaining"] += 1

    return Instance(k=k, categories=feature_info, agents=agents)


def legacy_find(feature_info: Dict[FeatureCategory, Dict[Feature, FeatureInfo]],
                agents: Dict[AgentId, Dict[FeatureCategory, Feature]], k: int) -> List[AgentId]:
    """Produce a single panel using the LEGACY algorithm, restarting until a panel is found."""

    columns_data = {agent_id: {} for agent_id in agents}
    while True:
        cat_copy = deepcopy(feature_info)
        agents_copy = deepcopy(agents)
        try:
            people_selected, new_output_lines = find_random_sample_legacy(cat_copy, agents_copy, columns_data, k, False,
                                                                          [])
        except SelectionError:
            continue
        # check we have reached minimum needed in all cats
        check_min_cat, new_output_lines = check_min_cats(cat_copy)
        if check_min_cat:
            return list(people_selected.keys())


def legacy_probabilities(instance: Instance, iterations: int, random_seed: int) -> ProbAllocation:
    """Estimate the probability allocation for LEGACY by drawing `iterations` many random panels in a row."""

    categories = instance.categories
    agents = instance.agents
    k = instance.k

    seed(random_seed)
    np.random.seed(random_seed)

    for category in categories:
        assert sum(fv_info["min"] for fv_info in categories[category].values()) <= k
        assert sum(fv_info["max"] for fv_info in categories[category].values()) >= k

    # For each agent, count how frequently she was selected among `iterations` many panels produced by LEGACY
    agent_appearance_counter = Counter()
    for i in range(iterations):
        if (i + 1) % 100 == 0:
            print(f"Running iteration {i + 1} out of {iterations}.")
        agent_appearance_counter.update(legacy_find(categories, agents, k))
    return ProbAllocation({agent_id: agent_appearance_counter[agent_id] / iterations for agent_id in agents})


def leximin_probabilities(instance: Instance) -> ProbAllocation:
    """Compute the exact probability allocation for LEXIMIN."""

    categories = instance.categories
    agents = instance.agents
    k = instance.k
    columns_data = {agent_id: {} for agent_id in agents}

    portfolio, output_probs, output_lines = find_distribution_leximin(categories, agents, columns_data, k, False, [])
    selection_probs = {agent_id: 0. for agent_id in agents}
    for panel, probability in zip(portfolio, output_probs):
        for agent_id in panel:
            selection_probs[agent_id] += probability
    return ProbAllocation(selection_probs)


@dataclass
class ProbAllocationStats:
    gini: float
    geometric_mean: float
    min: float


def compute_prob_allocation_stats(alloc: ProbAllocation, cap_for_geometric_mean: bool) -> ProbAllocationStats:
    """For an probability allocation, compute three measures of inequality: the Gini coefficient, the geometric mean,
    and the minimum selection probability.
    If `cap_for_geometric_mean` is True, probabilities below 1/10,000 are set to 1/10,000 before the calculation of the
    geometric mean to prevent the geometric mean from becoming zero. As described in the Methods section, we only give
    this advantage to the LEGACY benchmark.
    """
    n = len(alloc)
    k = round(sum(alloc.values()))

    # selection probabilities in increasing order
    sorted_probs = sorted(alloc.values())
    # Formulation for Gini coefficient adapted from:
    # Damgaard, C., & Weiner, J. (2000). Describing inequality in plant size or fecundity. Ecology, 81(4), 1139-1142.
    gini = sum((2 * i - n + 1) * prob for i, prob in enumerate(sorted_probs)) / (n * k)

    if cap_for_geometric_mean:
        capped_probs = [max(prob, 1 / 10000) for prob in alloc.values()]
        geometric_mean = gmean(capped_probs)
    else:
        geometric_mean = gmean(sorted_probs)

    mini = min(sorted_probs)

    return ProbAllocationStats(gini=gini, geometric_mean=geometric_mean, min=mini)


def upper_confidence_bound(num_trials: int, sample_proportion: float) -> float:
    """For a given sample size and observed sample proportion of a binomial variable, compute the 99th percentile of the
    Jeffrey’s prior updated with the observation
    (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Jeffreys_interval).
    """
    num_successes = round(sample_proportion * num_trials)
    if num_successes == num_trials:
        return 1.
    else:
        num_failures = num_trials - num_successes
        return beta.ppf(.99, .5 + num_successes, .5 + num_failures)


def run_legacy_or_retrieve(instance_name: str, instance: Instance, resample: bool) -> ProbAllocation:
    """Run the LEGACY algorithm or, if it has been run before, look up the result from disk. By setting `resample` to
    True, a second sample is taken with a different random seed.
    """
    if not resample:
        allocation_file = Path("distributions", f"{instance_name}_{instance.k}_legacy_first.pickle")
        random_seed = 0
    else:
        allocation_file = Path("distributions", f"{instance_name}_{instance.k}_legacy_second.pickle")
        random_seed = 1

    if allocation_file.exists():
        with open(allocation_file, "rb") as file:
            alloc = load(file)
    else:
        alloc = legacy_probabilities(instance, 10000, random_seed=random_seed)
        with open(allocation_file, "wb") as file:
            dump(alloc, file)

    assert len(alloc) == len(instance.agents)
    return alloc


def run_leximin_or_retrieve(instance_name: str, instance: Instance) -> ProbAllocation:
    """Run the LEXIMIN algorithm or, if it has been run before, look up the result from disk."""

    allocation_file = Path("distributions", f"{instance_name}_{instance.k}_leximin.pickle")

    if allocation_file.exists():
        with open(allocation_file, "rb") as file:
            alloc = load(file)
    else:
        alloc = leximin_probabilities(instance)
        with open(allocation_file, "wb") as file:
            dump(alloc, file)

    assert len(alloc) == len(instance.agents)
    return alloc


def plot_probability_allocations(instance_name: str, instance: Instance, legacy_alloc: ProbAllocation,
                                 leximin_alloc: ProbAllocation, leximin_min: float, share_below_leximin_min: float) \
        -> Path:
    """Produce line graph comparing the probability allocations of different algorithms on the same instance.
    For each algorithm, the selection probabilities are sorted in order of increasing selection probability.
    Plot is created at ./analysis/[instance_name]_[k]_prob_allocs.pdf and a table with raw data for the plot is created
    at ./analysis/[instance_name]_[k]_prob_allocs_data.csv . Returns the plath to the plot.
    """
    agents = instance.agents
    n = len(agents)
    k = instance.k

    # fictitious probability allocation where each agent receives equal probability k/n
    equalized_alloc = {agent_id: k / n for agent_id in agents}

    data = []
    plot_data = []
    for algorithm, alloc in [("Legacy", legacy_alloc), ("LexiMin", leximin_alloc), ("k/n", equalized_alloc)]:
        # Go through agents in order of increasing selection probability
        sorted_agents = sorted(agents, key=lambda agent_id: alloc[agent_id])
        for i, agent_id in enumerate(sorted_agents):
            data.append({"algorithm": algorithm, "percentile of pool members": i / n * 100,
                         "selection probability": alloc[agent_id]})
            if i == n - 1:  # needed to draw the last step
                plot_data.append({"algorithm": algorithm, "percentile of pool members": 100,
                                  "selection probability": alloc[agent_id]})

    output_directory = Path("analysis")
    df = pd.DataFrame(data)
    df.to_csv(output_directory / f"{instance_name}_{k}_prob_allocs_data.csv", index=False)
    df = pd.DataFrame(data + plot_data)

    max_percentile = 100
    restricted_df = df[df['percentile of pool members'] <= max_percentile]
    fig = sns.relplot(data=restricted_df, x="percentile of pool members",
                      y='selection probability', hue="algorithm", hue_order=("Legacy", "LexiMin", "k/n"), kind="line",
                      drawstyle='steps-post', height=6, aspect=1.8, legend=False)
    fig.set_axis_labels("percentile of pool members (by selection probability)", "selection probability")

    plt.xlim(left=0.0, right=max_percentile)
    plt.ylim(bottom=0.0)

    # confidence intervals for LEGACY probability allocation
    ax = fig.ax
    restricted_df = restricted_df[restricted_df["algorithm"] == "Legacy"]
    restricted_df.sort_values(by="percentile of pool members", inplace=True)
    times_selected = (10000 * restricted_df["selection probability"]).round()
    # 99% Jeffreys intervals, i.e., the 0.5th and the 99.5th quantile of Beta(1/2 + successes, 1/2 + failures), with
    # exceptions when the number of successes or failures is 0:
    # Brown, L. D., Cai, T. T. & DasGupta, A. Interval estimation for a binomial proportion. Statistical science 16,
    # 101–117 (2001).
    lower = beta.ppf(.005, .5 + times_selected, .5 + (10000 - times_selected))
    lower = np.where(times_selected == 0, 0., lower)
    upper = beta.ppf(.995, .5 + times_selected, .5 + (10000 - times_selected))
    upper = np.where(times_selected == 10000, 1., upper)
    ax.fill_between(restricted_df["percentile of pool members"], lower, upper,
                    alpha=0.25, edgecolor='none', facecolor="#2077B4", step="post")

    plt.legend((ax.lines[0], fig.ax.lines[1]), ('LEGACY', 'LEXIMIN'), fontsize=12)

    ax.lines[2].set_linestyle("dotted")
    ax.lines[2].set_color("g")
    ax.set_xlim(left=0.)
    ax.set_ylim(bottom=0)

    # equalized probability label, minimum probability line, and rectangle
    ax.text(1, 1.03 * k / n, 'equalized probability (k/n)', color='g')
    ax.axhline(y=leximin_min, xmin=0, xmax=max_percentile, linestyle='--', color='black', linewidth=0.5)
    rect = patches.Rectangle((0, 0), share_below_leximin_min * 100, leximin_min, edgecolor='none', facecolor="black",
                             alpha=0.1)
    ax.add_patch(rect)

    ax.set_title(f"Probability allocations on instance {instance_name}")

    plot_path = output_directory / f"{instance_name}_{k}_prob_allocs.pdf"
    fig.savefig(plot_path)
    return plot_path


def compute_ratio_products(instance: Instance) -> Dict[AgentId, float]:
    """For each agent in the pool, compute her ratio product, a quantification of how overrepresented her features are
    in the pool relative to the quotas.
    """
    feature_counter = Counter()
    for pers in instance.agents.values():
        feature_counter.update(pers.items())

    representation_ratios: Dict[Tuple[FeatureCategory, Feature], float]
    representation_ratios = {}
    for (feature, value), count in feature_counter.items():
        pool_share = count / len(instance.agents)
        quota_midpoint = (instance.categories[feature][value]["min"] + instance.categories[feature][value]["max"]) / 2
        quota_panel_share = quota_midpoint / instance.k
        representation_ratios[(feature, value)] = pool_share / quota_panel_share

    ratio_products = {}
    for agent_id, agent in instance.agents.items():
        ratio_products[agent_id] = reduce(operator.mul, (representation_ratios[cat_feature]
                                                         for cat_feature in agent.items()), 1.)
    return ratio_products


def plot_ratio_products(instance_name: str, instance: Instance, allocation: ProbAllocation,
                        ratio_products: Dict[AgentId, float]):
    """Produce scatter plot showing the relationship between an agent's ratio product and her selection probability in
    the given probability allocation.
    Plot is created at ./analysis/[instance_name]_[k]_ratio_product.pdf and a table with raw data for the plot is
    created at ./analysis/[instance_name]_[k]_ratio_product_data.csv . Returns the plath to the plot.
    """
    data = []
    for agent_id in instance.agents:
        data.append({"ratio product": ratio_products[agent_id], "selection probability": allocation[agent_id]})

    output_directory = Path("analysis")
    df = pd.DataFrame(data)
    df.to_csv(output_directory / f"{instance_name}_{instance.k}_ratio_product_data.csv", index=False)
    fig = sns.relplot(data=df, x="ratio product", y="selection probability", kind="scatter", facecolor='none',
                      edgecolor='blue', alpha=0.8)
    max_ratio_product = max(df["ratio product"])
    fig.ax.set_xlim(-.02 * max_ratio_product, 1.02 * max_ratio_product)

    plot_path = output_directory / f"{instance_name}_{instance.k}_ratio_product.pdf"
    fig.savefig(plot_path)

    return plot_path


def has_features(agent: Dict[FeatureCategory, Feature], *feature_values):
    for category, feature in feature_values:
        if agent[category] != feature:
            return False
    return True


def get_quota_share(instance: Instance, *feature_values) -> float:
    quota_share = 1
    for category, feature in feature_values:
        mean_quota = (instance.categories[category][feature]["min"] + instance.categories[category][feature]["max"]) / 2
        quota_share *= mean_quota / instance.k
    return quota_share


def plot_intersectional_representation(instance_name: str, instance: Instance, legacy_alloc: ProbAllocation,
                                       leximin_alloc: ProbAllocation):
    """If the instance directory contains an additional table `intersections.csv`, which includes the population shares
    for a range of 2-feature intersections, produce a plot and report mean squared errors of the different panel/pool/
    quota shares. If table is not available, return None.
    Plot is created at ./analysis/[instance_name]_[k]_intersections.pdf .
    Returns a dictionary of mean squared errors (MSEs) mapping from pairs of shares (e.g.,
    `("panel share LEGACY" - "population share")`) to the corresponding MSE.
    """
    input_directory = Path("data", f"{instance_name}_{instance.k}")
    assert input_directory.is_dir()
    intersection_path = input_directory / "intersections.csv"
    if not intersection_path.exists():
        return None

    intersection_data = []
    with open(intersection_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for entry in reader:
            catfeature1 = entry["category 1"], entry["feature 1"]
            catfeature2 = entry["category 2"], entry["feature 2"]
            population_share = float(entry["population share"])
            panel_share_legacy = sum(prob for pers, prob in legacy_alloc.items() if
                                     has_features(instance.agents[pers], catfeature1, catfeature2)) / instance.k
            panel_share_leximin = sum(prob for pers, prob in leximin_alloc.items() if
                                      has_features(instance.agents[pers], catfeature1, catfeature2)) / instance.k
            pool_share = (sum(1 for agent in instance.agents.values() if has_features(agent, catfeature1, catfeature2))
                          / len(instance.agents))
            quota_share = get_quota_share(instance, catfeature1, catfeature2)

            intersection_data.append({"population share": population_share, "panel share LEGACY": panel_share_legacy,
                                      "panel share LEXIMIN": panel_share_leximin, "pool share": pool_share,
                                      "quota share": quota_share})

    df = pd.DataFrame(intersection_data)
    diff_pairs = (("panel share LEXIMIN", "population share"), ("panel share LEGACY", "population share"),
                  ("panel share LEXIMIN", "pool share"), ("panel share LEGACY", "pool share"),
                  ("panel share LEXIMIN", "quota share"), ("panel share LEGACY", "quota share"),
                  ("panel share LEXIMIN", "panel share LEGACY"))
    errors = {}
    for share1, share2 in diff_pairs:
        diff_name = share1 + " - " + share2
        df[diff_name] = df[share1] - df[share2]
        errors[(share1, share2)] = (df[diff_name]**2).mean()

    sns.set_style("whitegrid")
    scale = max(df["panel share LEXIMIN - population share"].abs().max(),
                df["panel share LEGACY - population share"].abs().max())
    scale = ceil(scale * 100) / 100

    f = sns.jointplot(data=df, x="panel share LEXIMIN - population share", y="panel share LEGACY - population share")
    f.ax_joint.set_xlim(-scale, scale)
    f.ax_joint.set_ylim(-scale, scale)
    output_path = Path("analysis", f"{instance_name}_{instance.k}_intersections.pdf")
    f.savefig(output_path)

    return errors


def analyze_instance(instance_name: str, instance: Instance, skip_timing: bool = False):
    """Run a full analysis of LEGACY and LEXIMIN on a given instance."""

    legacy_alloc_first_sample = run_legacy_or_retrieve(instance_name, instance, False)
    legacy_alloc_second_sample = run_legacy_or_retrieve(instance_name, instance, True)
    leximin_alloc = run_leximin_or_retrieve(instance_name, instance)

    k = instance.k
    n = len(instance.agents)

    legacy_stats = compute_prob_allocation_stats(legacy_alloc_first_sample, True)
    leximin_stats = compute_prob_allocation_stats(leximin_alloc, False)

    analysis_log = open(Path("analysis", f"{instance_name}_{k}_statistics.txt"), "w", encoding="utf-8")

    def log(*info):
        print(*info)
        analysis_log.write("\t".join(str(msg) for msg in info) + "\n")

    log("instance:", instance_name)
    log("pool size n:", n)
    log("panel size k:", k)
    log("# quota categories:", len(instance.categories))
    log("mean selection probability k/n:", f"{k / n:.1%}")

    # select one agent least frequently selected in the first set of 10,000 randomly drawn panels
    legacy_first_minimizer = min(instance.agents.keys(), key=lambda agent_id: legacy_alloc_first_sample[agent_id])
    first_minimizer_second_prob = legacy_alloc_second_sample[legacy_first_minimizer]

    log("LEGACY minimum probability:", f"≤ {upper_confidence_bound(10000, first_minimizer_second_prob):.2%} (99% upper "
                                       f"confidence bound based on Jeffreys interval for a binomial parameter, "
                                       f"calculated from sample proportion {first_minimizer_second_prob:.4f} and "
                                       f"sample size 10,000)")
    log("LEXIMIN minimum probability (exact):", f"{leximin_stats.min:.1%}")

    log("gini coefficient of LEGACY:", f"{legacy_stats.gini:.1%}")
    log("gini coefficient of LEXIMIN:", f"{leximin_stats.gini:.1%}")
    log("geometric mean of LEGACY:", f"{legacy_stats.geometric_mean:.1%}")
    log("geometric mean of LEXIMIN:", f"{leximin_stats.geometric_mean:.1%}")
    share_below_leximin_min = sum(1 for p in legacy_alloc_first_sample.values() if p < leximin_stats.min) / n
    log("share selected by LEGACY with probability below LEXIMIN minimum selection probability:",
        f"{share_below_leximin_min:.1%}")

    prob_alloc_plot_path = plot_probability_allocations(instance_name, instance, legacy_alloc_first_sample,
                                                        leximin_alloc, leximin_stats.min, share_below_leximin_min)
    log(f"Plot of probability allocation created at {prob_alloc_plot_path}.")

    ratio_products = compute_ratio_products(instance)
    ratio_product_plot_path = plot_ratio_products(instance_name, instance, legacy_alloc_first_sample, ratio_products)
    log(f"Plot of ratio products created at {ratio_product_plot_path}.")

    intersection_errors = plot_intersectional_representation(instance_name, instance, legacy_alloc_first_sample,
                                                             leximin_alloc)
    if intersection_errors is not None:
        for (share1, share2), error in intersection_errors.items():
            log(f"MSE({share1}, {share2})", f"{error:.2e}")

    if skip_timing:
        log("Skip timing.")
        return

    analysis_log.flush()
    timings = []
    for i in range(3):
        start = time()
        leximin_probabilities(instance)
        end = time()
        timings.append(end - start)
        print(f"Run {i + 1}/3 of LEXIMIN took {end - start:.1f} seconds.")
    timings.sort()
    log(f"Out of 3 runs, LEXIMIN took a median running time of {timings[1]:.1f} seconds.")

    analysis_log.close()


if __name__ == '__main__':
    valid_inputs = []
    errors = []
    for subdir in sorted(Path("data").iterdir()):
        if subdir.is_dir():
            if not "_" in subdir.name:
                errors.append((subdir.name, "directory name does not end in underscore followed by a panel size"))
                continue
            data_name, k_string = subdir.name.rsplit("_", 1)
            try:
                k = int(k_string)
            except ValueError:
                errors.append((subdir.name, "directory name does not end in underscore followed by a panel size"))
                continue
            cat_file = subdir.joinpath("categories.csv")
            if not cat_file.exists():
                errors.append((subdir.name, "directory does not contain file 'categories.csv'"))
                continue
            resp_file = subdir.joinpath("respondents.csv")
            if not resp_file.exists():
                errors.append((subdir.name, "directory does not contain file 'respondents.csv'"))
                continue
            valid_inputs.append((data_name, k))
    epilog = []
    if len(valid_inputs) > 0:
        epilog.append("Valid input combinations of instance_name and panel_size (based on subdirectories of ./data):")
        for data_name, k in valid_inputs:
            epilog.append("    " + (data_name + ", ").ljust(15) + str(k))
    else:
        epilog.append("Based on subdirectories of ./data, no instances were found.")
    if len(errors) > 0:
        epilog.append("There were additional subdirectories, but some problem prevents them from being used:")
        for dir_name, error in errors:
            epilog.append("    " + (dir_name + ": ").ljust(15) + error)

    parser = ArgumentParser(description=("Analyze a given instance, creating the information given in all tables of \n"
                                         "the paper and creating versions of the plots as well as raw data for these \n"
                                         "plots. Produced information is logged to the file \n"
                                         "./analysis/[INSTANCE_NAME]_[PANEL_SIZE]_statistics.txt ."),
                            epilog="\n".join(epilog),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('instance_name', type=str,
                        help='name of the instance (str)')
    parser.add_argument('panel_size', type=int,
                        help='panel size (int)')
    parser.add_argument('--skiptiming', action="store_true", help='do not time three runs of LEXIMIN')

    args = parser.parse_args()

    instance_name = args.instance_name
    k = args.panel_size
    skip_timing = args.skiptiming

    if (instance_name, k) not in valid_inputs:
        print("Input does not specify a valid combination of instance name and panel size.")
        print("\n".join(epilog))
    else:
        input_directory = Path("data", f"{instance_name}_{k}")
        instance = read_instance(input_directory / "categories.csv", input_directory / "respondents.csv", k)
        analyze_instance(instance_name, instance, skip_timing)
