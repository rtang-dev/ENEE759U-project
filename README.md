# Automated ILP Scheduling Tool

This tool automatically takes a DFG (Data Flow Graph) in edgelist format as input and generates a schedule under various constraints. It can:

1. Minimize memory given a latency constraint (L).
2. Minimize latency given a memory constraint (M).
3. Perform a Pareto-optimal analysis to explore the trade-off between latency and memory.

## Features
- **Input:** A DAG in edgelist format. Each edge includes a memory cost (weight).
- **Objectives:**  
  - *Minimize Memory under Latency L*: Given L, minimize the peak memory usage.  
  - *Minimize Latency under Memory M*: Given M, minimize the maximum latency.  
  - *Pareto Analysis*: Generate a Pareto frontier of latency vs memory usage.

## Requirements
- Python 3+
- `networkx`
- `pulp` (with a compatible ILP solver like HiGHS or GLPK)
- `matplotlib` (for Pareto plot)

Install dependencies:
```bash
pip install networkx pulp matplotlib
```
Install HiGHS:
Use the following commands.
```bash
wget https://github.com/ERGO-Code/HiGHS/releases/download/v1.6.1/highs-1.6.1-linux64.zip
unzip highs-1.6.1-linux64.zip
chmod +x highs
mv highs /usr/local/bin/
```
Or download precompiled binaries.
## Usage

python auto_schedule.py -g=<edgelist_file> [OPTIONS]

### Options:

- `-l, --latency <int>`: Fixed latency, minimize memory.
- `-m, --memory <int>`: Fixed memory, minimize latency.
- `-p, --pareto`: Perform Pareto-optimal analysis.
- `-o, --output <filename>`: Output Pareto plot filename.

### Examples:
- Minimize memory with latency=4:
```bash
python auto_schedule.py -l=4 -g=rand_DFG_s10_1.edgelist
```
- Minimize latency with memory=10:
```bash
python auto_schedule.py -m=10 -g=rand_DFG_s10_1.edgelist
```

- Pareto analysis:
```bash
python auto_schedule.py -p -g=rand_DFG_s50_7.edgelist -o=pareto_s50_1.png
```

## Output

    If feasible, prints optimal result and start times of each node;
    If not, prints the problems is not feasible.
    For Pareto analysis, saves a .png showing the frontier.
