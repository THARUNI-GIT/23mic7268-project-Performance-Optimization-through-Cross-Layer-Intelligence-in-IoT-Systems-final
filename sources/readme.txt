================================================================================
  ACLIF — Autonomous Cross-Layer Intelligence Framework
  Performance Optimization through Cross-Layer Intelligence in IoT Systems
================================================================================

AUTHOR
------
  Name   : B. Tharuni Sri Sai
  ORCID  : 0009-0004-9561-8985
  Email  : tharuni.23mic7268@vitapstudent.ac.in
  GitHub : https://github.com/THARUNI-GIT/23mic7268-project-Performance-Optimization-through-Cross-Layer-Intelligence-in-IoT-Systems
  Gmail  : tharuni.23mic7268@vitapstudent.ac.in
  Affiliation : School of Computer Science and Engineering
                VIT-AP University, Amaravati, Andhra Pradesh, India

================================================================================
REPOSITORY CONTENTS
================================================================================

  figures/                   — All 12 result PNG plots (res1–res12)
  tex/res1.tex ... res12.tex — Standalone LaTeX result files (compile individually)
  cc/res1.cc  ... res12.cc   — NS3 v3.38 C++ simulation scripts
  execution.ipynb            — DQN training + NS3 launcher (ML/DL execution)
  results.ipynb              — All 12 result reproduction plots
  readme.txt                 — This file

================================================================================
RESULT FILE MAPPING
================================================================================

  res1   — Average End-to-End Delay vs Node Density
  res2   — Per-Round Energy Consumption vs Simulation Rounds
  res3   — Network Throughput vs Packet Generation Rate
  res4   — MAC Collision Rate vs Node Density
  res5   — Communication Overhead vs Packet Rate
  res6   — Computational Overhead per Decision Cycle vs Node Density
  res7   — Delay-Energy Performance Tradeoff (Pareto scatter)
  res8   — Ablation Study: Individual Action Contributions
  res9   — Comparative Performance Radar Chart
  res10  — DQN Training Convergence and Epsilon Decay
  res11  — Sensitivity Analysis: Path-Loss Exponent alpha vs Delay
  res12  — Mobility Impact: Static / Constant Velocity / Random Waypoint

================================================================================
SIMULATION ENVIRONMENT
================================================================================

  Simulator  : NS3 discrete-event simulator, version 3.38
  PHY layer  : IEEE 802.15.4, log-distance path-loss (alpha=3.5)
  MAC layer  : CSMA/CA, CW_min=16, CW_max=256
  Nodes      : 100 sensor nodes, 200x200 m^2 area
  Traffic    : Poisson, lambda=5 pkt/s, 512-byte payload
  Runs       : 10 per configuration (different random seeds)
  CI         : 95% Student-t with 9 degrees of freedom
  Sim time   : 300 seconds per run
  Energy     : First-order radio model, E_init=2.0 J/node

  Baselines compared:
    - E-GLBR  : Genetic Algorithm Routing (Benelhouri et al., 2023)
    - FDRL    : Federated Deep RL Routing (Suresh et al., 2024)
    - FACSO   : Fuzzy Adaptive Cross-Layer Scheduling (Yang et al., 2024)
    - EEHCT   : Energy-Efficient Clustering with TDMA (Chaurasiya et al., 2023)
    - IPSO    : Improved PSO Multi-Hop Routing (Zhang et al., 2025)

================================================================================
HOW TO RUN NS3 SIMULATIONS
================================================================================

  1. Install NS3 v3.38:
       git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3.38
       cd ns-3.38 && git checkout ns-3.38
       ./ns3 configure --enable-examples && ./ns3 build

  2. Copy CC files to scratch:
       cp cc/res*.cc ~/ns-3.38/scratch/

  3. Run a simulation (example: Result 1, 100 nodes):
       cd ~/ns-3.38
       ./ns3 run "scratch/res1 --nodeCount=100 --seed=1"

  4. Sweep over node densities (bash loop):
       for N in 20 40 60 80 100; do
         ./ns3 run "scratch/res1 --nodeCount=$N --seed=42"
       done

  5. Parse output CSVs and plot using results.ipynb

================================================================================
HOW TO COMPILE LaTeX RESULT FILES
================================================================================

  Each resN.tex is a standalone compilable document.
  Requires: pdflatex, booktabs, amsmath, graphicx packages.

  Single file:
    cd tex/
    pdflatex res1.tex

  All 12 files:
    for i in $(seq 1 12); do pdflatex res${i}.tex; done

  Note: Figures are referenced as ../figures/figN.png (relative path).
  Copy or symlink the figures/ directory alongside tex/ when compiling.

================================================================================
HOW TO RUN JUPYTER NOTEBOOKS
================================================================================

  Requirements:
    pip install torch numpy pandas matplotlib scipy tqdm jupyter

  Launch:
    jupyter notebook execution.ipynb   # ML/DL training
    jupyter notebook results.ipynb     # Result reproduction

  execution.ipynb:
    - Defines DQN architecture (128→64 hidden, 15-dim input, 81-dim output)
    - Trains agent for 300 episodes on Python IoT surrogate
    - Saves model to models/aclif_dqn.pth
    - Includes NS3 subprocess launcher for full simulation integration

  results.ipynb:
    - Reproduces all 12 result figures from paper simulation data
    - Outputs figures to figures/resN_*.png
    - Exports summary CSV: figures/results_summary.csv

================================================================================
ACLIF KEY RESULTS SUMMARY
================================================================================

  Metric                        ACLIF       Best Baseline   Improvement
  ─────────────────────────────────────────────────────────────────────
  Avg Delay @ 100 nodes (ms)    91.5        118.2 (IPSO)    22.6%
  Energy per round (J)          0.0291      0.0379 (IPSO)   23.2%
  Throughput @ λ=10 (kbps)      83.7        69.8  (IPSO)    19.9%
  Collision @ 100 nodes (%)     8.3         14.9  (IPSO)    44.3%
  DQN convergence               ~280 episodes
  State vector dimension        15
  Action space size             81 (3^4 joint actions)
  Control interval              500 ms

================================================================================
CITATION
================================================================================

  If you use this code or results, please cite:

  B. Tharuni Sri Sai, "Performance Optimization through Cross-Layer Intelligence
  in IoT Systems," VIT-AP University, Amaravati, India, 2026.
  ORCID: 0009-0004-9561-8985

================================================================================
LICENSE
================================================================================

  For academic and research use only.
  Contact: tharuni.23mic7268@vitapstudent.ac.in

================================================================================
