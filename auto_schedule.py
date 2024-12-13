#!/usr/bin/env python3
"""
Automated ILP Scheduling Tool for DFGs
Optimized for large L by restricting variable ranges and using a tighter M.

Now using the corrected inequality for memory usage:
For an edge (i,j) to be "alive" at step l, we need: t_i ≤ l ≤ t_j - 1.

We define:
- A_{i,l}=1 if t_i ≤ l, defined only for l in [asap[i], alap[i]].
- B_{j,l}=1 if t_j - 1 ≥ l, defined only for l in [1, alap[j]].

For y_{ij,l}, we consider all l in [1, L].
- If l<asap[i], t_i ≤ l can't hold → y_{ij,l}=0.
- If l>alap[j], t_j -1≥ l can't hold → y_{ij,l}=0.
- If asap[i] ≤ l ≤ alap[i] and 1 ≤ l ≤ alap[j], we can form constraints with A_{i,l} and B_{j,l}.
- If l>alap[i], t_i ≤ l always holds (A=1). If l ≤ alap[j], B exists; if l>alap[j], y=0.
- If l<asap[i], y=0.
"""

import argparse
import networkx as nx
import random
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpBinary,
    LpInteger,
    LpStatus,
    HiGHS_CMD,
)
import matplotlib.pyplot as plt
import sys


def load_graph(edgelist_file):
    graph = nx.read_edgelist(
        edgelist_file,
        nodetype=int,
        data=(("weight", int),),
        create_using=nx.DiGraph(),
    )
    return graph


def parse_arguments():
    parser = argparse.ArgumentParser(description="Automated ILP Scheduling Tool")
    parser.add_argument("-l", "--latency", type=int, help="Latency constraint L")
    parser.add_argument("-m", "--memory", type=int, help="Memory constraint M")
    parser.add_argument("-g", "--graph", type=str, required=True, help="Path to the edgelist file")
    parser.add_argument("-p", "--pareto", action="store_true", help="Perform Pareto-optimal analysis")
    parser.add_argument("-o", "--output", type=str, help="Output filename for Pareto frontier plot")
    args = parser.parse_args()
    return args


def compute_ASAP_times(graph):
    topo = list(nx.topological_sort(graph))
    asap = {v: 1 for v in topo}
    for v in topo:
        preds = list(graph.predecessors(v))
        if preds:
            asap[v] = max(asap[p] + 1 for p in preds)
    return asap


def compute_ALAP_times(graph, max_step):
    rev = graph.reverse()
    topo_rev = list(nx.topological_sort(rev))
    alap = {v: max_step for v in topo_rev}
    for v in topo_rev:
        succs = list(graph.successors(v))
        if succs:
            alap[v] = min(alap[s] - 1 for s in succs)
    return alap


def solve_problem(problem):
    solver = HiGHS_CMD(msg=True, timeLimit=1800, threads=16, options=["--presolve=on", "--parallel=on"])
    result_status = problem.solve(solver)
    return result_status


def extract_solution(x):
    start_times = {}
    for (i, l), var in x.items():
        if var.varValue > 0.5:
            start_times[i] = l
    return start_times


def check_feasibility(problem):
    status = LpStatus[problem.status]
    if status == "Infeasible":
        return False
    elif status == "Optimal":
        return True
    return False


def minimize_memory(graph, L, asap, alap):
    nodes = list(graph.nodes())
    edges = list(graph.edges(data=True))

    min_asap = min(asap.values())
    max_alap = max(alap.values())
    M = (max_alap - min_asap) + 1

    problem = LpProblem("MinimizeMemoryUnderLatency", LpMinimize)

    x = {(i, l): LpVariable(f"x_{i}_{l}", cat=LpBinary) for i in nodes for l in range(asap[i], alap[i] + 1)}
    m_max = LpVariable("m_max", lowBound=0, cat=LpInteger)

    # Each task once
    for i in nodes:
        problem += lpSum([x[(i, ll)] for ll in range(asap[i], alap[i] + 1)]) == 1

    # Precedence
    for i, j, data in edges:
        problem += (
            lpSum([ll * x[(j, ll)] for ll in range(asap[j], alap[j] + 1)])
            >= lpSum([ll * x[(i, ll)] for ll in range(asap[i], alap[i] + 1)]) + 1
        )

    # Latency
    for i in nodes:
        problem += lpSum([ll * x[(i, ll)] for ll in range(asap[i], alap[i] + 1)]) <= L

    A = {}
    B = {}
    y = {}

    # A_{i,l}
    for i in nodes:
        t_i_expr = lpSum([ll * x[(i, ll)] for ll in range(asap[i], alap[i] + 1)])
        for step in range(asap[i], alap[i] + 1):
            A[(i, step)] = LpVariable(f"A_{i}_{step}", cat=LpBinary)
            problem += t_i_expr <= step + M * (1 - A[(i, step)])
            problem += t_i_expr >= step - M * A[(i, step)] + 1

    # B_{j,l}
    for j in nodes:
        t_j_expr = lpSum([ll * x[(j, ll)] for ll in range(asap[j], alap[j] + 1)])
        for step in range(1, alap[j] + 1):
            B[(j, step)] = LpVariable(f"B_{j}_{step}", cat=LpBinary)
            problem += step <= (t_j_expr - 1) + M * (1 - B[(j, step)])
            problem += step >= (t_j_expr - 1) - M * B[(j, step)] + 1

    # y_{ij,l}
    for i, j, data in edges:
        for step in range(1, L + 1):
            if step < asap[i]:
                y[(i, j, step)] = LpVariable(f"y_{i}_{j}_{step}", cat=LpBinary)
                problem += y[(i, j, step)] == 0
            elif step > alap[j]:
                y[(i, j, step)] = LpVariable(f"y_{i}_{j}_{step}", cat=LpBinary)
                problem += y[(i, j, step)] == 0
            else:
                if step <= alap[j]:
                    if (j, step) in B:
                        y[(i, j, step)] = LpVariable(f"y_{i}_{j}_{step}", cat=LpBinary)
                        if asap[i] <= step <= alap[i]:
                            problem += y[(i, j, step)] >= A[(i, step)] + B[(j, step)] - 1
                            problem += y[(i, j, step)] <= A[(i, step)]
                            problem += y[(i, j, step)] <= B[(j, step)]
                        elif step > alap[i]:
                            problem += y[(i, j, step)] >= B[(j, step)]
                            problem += y[(i, j, step)] <= B[(j, step)]
                    else:
                        y[(i, j, step)] = LpVariable(f"y_{i}_{j}_{step}", cat=LpBinary)
                        problem += y[(i, j, step)] == 0

    steps_with_y = set(l for (ii, jj, l) in y.keys())
    for step in sorted(steps_with_y):
        problem += lpSum([y[(ii, jj, step)] * d["weight"] for (ii, jj, d) in edges if (ii, jj, step) in y]) <= m_max

    problem += m_max, "Minimize_m_max"

    return problem, x, m_max


def minimize_latency(graph, M_val, asap, alap):
    nodes = list(graph.nodes())
    edges = list(graph.edges(data=True))

    L = max(alap.values())

    problem = LpProblem("MinimizeLatencyUnderMemory", LpMinimize)

    x = {(i, l): LpVariable(f"x_{i}_{l}", cat=LpBinary) for i in nodes for l in range(asap[i], alap[i] + 1)}

    L_var = LpVariable("L_var", lowBound=0, cat=LpInteger)

    min_asap = min(asap.values())
    max_alap = max(alap.values())
    M = (max_alap - min_asap) + 1

    # Each task once
    for i in nodes:
        problem += lpSum([x[(i, ll)] for ll in range(asap[i], alap[i] + 1)]) == 1

    # Precedence
    for i, j, data in edges:
        problem += (
            lpSum([ll * x[(j, ll)] for ll in range(asap[j], alap[j] + 1)])
            >= lpSum([ll * x[(i, ll)] for ll in range(asap[i], alap[i] + 1)]) + 1
        )

    # L_var ≥ t_i
    for i in nodes:
        problem += L_var >= lpSum([ll * x[(i, ll)] for ll in range(asap[i], alap[i] + 1)])

    A = {}
    B = {}
    y = {}

    # A_{i,l}
    for i in nodes:
        t_i_expr = lpSum([ll * x[(i, ll)] for ll in range(asap[i], alap[i] + 1)])
        for step in range(asap[i], alap[i] + 1):
            A[(i, step)] = LpVariable(f"A_{i}_{step}", cat=LpBinary)
            problem += t_i_expr <= step + M * (1 - A[(i, step)])
            problem += t_i_expr >= step - M * A[(i, step)] + 1

    # B_{j,l}
    for j in nodes:
        t_j_expr = lpSum([ll * x[(j, ll)] for ll in range(asap[j], alap[j] + 1)])
        for step in range(1, alap[j] + 1):
            B[(j, step)] = LpVariable(f"B_{j}_{step}", cat=LpBinary)
            problem += step <= (t_j_expr - 1) + M * (1 - B[(j, step)])
            problem += step >= (t_j_expr - 1) - M * B[(j, step)] + 1

    # y_{ij,l}
    for i, j, data in edges:
        for step in range(1, L + 1):
            if step < asap[i]:
                y[(i, j, step)] = LpVariable(f"y_{i}_{j}_{step}", cat=LpBinary)
                problem += y[(i, j, step)] == 0
            elif step > alap[j]:
                y[(i, j, step)] = LpVariable(f"y_{i}_{j}_{step}", cat=LpBinary)
                problem += y[(i, j, step)] == 0
            else:
                if step <= alap[j]:
                    if (j, step) in B:
                        y[(i, j, step)] = LpVariable(f"y_{i}_{j}_{step}", cat=LpBinary)
                        if asap[i] <= step <= alap[i]:
                            problem += y[(i, j, step)] >= A[(i, step)] + B[(j, step)] - 1
                            problem += y[(i, j, step)] <= A[(i, step)]
                            problem += y[(i, j, step)] <= B[(j, step)]
                        elif step > alap[i]:
                            problem += y[(i, j, step)] >= B[(j, step)]
                            problem += y[(i, j, step)] <= B[(j, step)]
                    else:
                        y[(i, j, step)] = LpVariable(f"y_{i}_{j}_{step}", cat=LpBinary)
                        problem += y[(i, j, step)] == 0

    steps_with_y = set(l for (ii, jj, l) in y.keys())
    for step in sorted(steps_with_y):
        problem += lpSum([y[(ii, jj, step)] * d["weight"] for (ii, jj, d) in edges if (ii, jj, step) in y]) <= M_val

    problem += L_var, "Minimize_Latency"

    return problem, x, L_var


def plot_pareto_frontier(results, output_filename="pareto.png"):
    latencies = [r[0] for r in results]
    memories = [r[1] for r in results]

    plt.figure()
    plt.plot(memories, latencies, "o-", label="Pareto Frontier")
    plt.xlabel("Maximum Memory Usage")
    plt.ylabel("Latency")
    plt.title("Pareto-Optimal Frontier")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()


def pareto_optimal_analysis(graph, asap, alap, output_filename):
    results = set()
    min_latency = nx.dag_longest_path_length(graph, weight=None) + 1
    max_latency = min_latency + 10
    latency_step = 2

    print("Analyzing memory minimization under latency constraints:")
    for L_val in range(min_latency, max_latency + 1, latency_step):
        problem, x, m_max = minimize_memory(graph, L_val, asap, alap)
        solve_problem(problem)
        if check_feasibility(problem):
            max_memory = m_max.varValue
            print(f"Latency: {L_val}, Minimized Memory Usage: {max_memory}")
            results.add((L_val, max_memory))

    if results:
        memory_usages = [m for _, m in results]
        min_memory = int(min(memory_usages))
        max_memory = int(max(memory_usages))
    else:
        min_memory = 1
        max_memory = 2000

    memory_step = max(1, int((max_memory - min_memory) / 4))
    if memory_step == 0:
        memory_step = 1

    print("\nAnalyzing latency minimization under memory constraints:")
    for M in range(min_memory, max_memory + 1, memory_step):
        problem, x, L_var = minimize_latency(graph, M, asap, alap)
        solve_problem(problem)
        if check_feasibility(problem):
            latency = L_var.varValue
            print(f"Memory Constraint: {M}, Minimized Latency: {latency}")
            results.add((latency, M))

    results = list(results)
    pareto_front = []
    for current in results:
        dominated = False
        for other in results:
            if other == current:
                continue
            if other[0] <= current[0] and other[1] <= current[1] and (other[0] < current[0] or other[1] < current[1]):
                dominated = True
                break
        if not dominated:
            pareto_front.append(current)

    pareto_front = sorted(pareto_front, key=lambda x: x[1])
    if pareto_front:
        plot_pareto_frontier(pareto_front, output_filename=output_filename)
    else:
        print("No Pareto-optimal solutions found.")


def main():
    args = parse_arguments()
    graph = load_graph(args.graph)

    if not nx.is_directed_acyclic_graph(graph):
        print("The provided graph is not a DAG. Exiting.")
        sys.exit(1)

    if args.latency is not None:
        L_for_alap = args.latency
    else:
        length = nx.dag_longest_path_length(graph, weight=None)
        L_for_alap = length + len(graph.nodes()) + 100

    asap = compute_ASAP_times(graph)
    alap = compute_ALAP_times(graph, L_for_alap)

    if args.pareto:
        output_filename = args.output if args.output else "pareto.png"
        pareto_optimal_analysis(graph, asap, alap, output_filename=output_filename)
    elif args.latency is not None:
        problem, x, m_max = minimize_memory(graph, args.latency, asap, alap)
        solve_problem(problem)
        if check_feasibility(problem):
            print(f"Optimal Maximum Memory Usage: {m_max.varValue}")
            start_times = extract_solution(x)
            for node in sorted(start_times.keys()):
                print(f"Node {node}: starts at time {start_times[node]}")
        else:
            print("No feasible solution found.")
            sys.exit(1)
    elif args.memory is not None:
        problem, x, L_var = minimize_latency(graph, args.memory, asap, alap)
        solve_problem(problem)
        if check_feasibility(problem):
            print(f"Optimal Latency: {L_var.varValue}")
            start_times = extract_solution(x)
            for node in sorted(start_times.keys()):
                print(f"Node {node}: starts at time {start_times[node]}")
        else:
            print("No feasible solution found.")
            sys.exit(1)
    else:
        print("Please provide either a latency (-l) or memory (-m) constraint, or use -p for Pareto analysis.")
        sys.exit(1)


if __name__ == "__main__":
    main()
