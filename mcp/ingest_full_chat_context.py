import os
import sys
import json
import time

# Add MCP server path
mcp_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\geodesic_memory_mcp"
sys.path.insert(0, mcp_dir)
from server import GeodesicMemoryGraph, hash_text_to_phase_vector

transcript_path = r"C:\Users\Admin\.gemini\antigravity\brain\a6ace972-7a51-4875-b0fd-e9b9d6226f42\.system_generated\logs\transcript.jsonl"
report_path = r"C:\Users\Admin\.gemini\antigravity\brain\a6ace972-7a51-4875-b0fd-e9b9d6226f42\geodesic_spacetime_hysteresis_report.md"
repo_data_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\geodesic-symplectic-engine\data"

def main():
    print("=" * 80)
    print("FULL CONVERSATION CONTEXT INGESTION ENGINE")
    print("Physical Phase-Space Geodesic Ingestion with Supercritical Hysteresis Latches")
    print("=" * 80)

    graph = GeodesicMemoryGraph()
    t0 = time.perf_counter()

    # 1. Structured Knowledge Nodes representing the entire journey
    nodes_data = [
        {
            "concept": "Research Mission & Tsundere Dynamic Persona System",
            "content": "Collaborative research partnership between user and hyper-rigorous autistic tsundere scientist. Global instructions in GEMINI.md enforcing cold-hard terminal evidence, anti-sycophancy null-hypothesis testing, and 4-state dynamic tsundere spectrum (Debugging 90/10, Breakthrough 40/60, Banter 70/30, Late-Night 25/75).",
            "category": "meta_protocol",
            "links": []
        },
        {
            "concept": "Phase 1: Complex Wave Superposition & XOR 0-Hidden Layers",
            "content": "Verified single-neuron wave interference W_j * s_j(x) solving non-linear boundary XOR (100% accuracy) and concentric ring classification with 0 hidden layers in test_superposition_xor.py.",
            "category": "neural_architecture",
            "links": ["Research Mission & Tsundere Dynamic Persona System"]
        },
        {
            "concept": "Phase 2: Supercritical Pitchfork Hysteresis 100-Step Noise Latch",
            "content": "Discovered that internal non-linear feedback ds/dt = -s + tanh(w_self * s + I) operating past supercritical pitchfork bifurcation (w_self = 4.208 > 1) achieves 100% 1-bit memory retention across 100 continuous steps of white noise in test_hysteretic_neuron.py, while vanilla RNNs decayed to 50.5% (random chance).",
            "category": "memory_dynamics",
            "links": ["Phase 1: Complex Wave Superposition & XOR 0-Hidden Layers"]
        },
        {
            "concept": "Phase 3: Group Associative Emergence & Inverted-U Limit Cycles",
            "content": "Benchmarked 20-neuron associative network in test_hysteretic_group_emergence.py. Found inverted-U phase transition (w_self=0 -> 0%, w_self=0.8 -> 74%, w_self=2.5 -> 0% due to lockup) and autonomous Central Pattern Generator (CPG) Andronov-Hopf limit cycle oscillation.",
            "category": "group_dynamics",
            "links": ["Phase 2: Supercritical Pitchfork Hysteresis 100-Step Noise Latch"]
        },
        {
            "concept": "Phase 4: Kerr Spacetime Singularity Scaling vs 6,529-Param MLP",
            "content": "Evaluated on N=5,000 points with noise in benchmark_kerr_vs_mlp_scaling.py. 9-parameter single Kerr neuron beat a 6,529-parameter MLP (97.96% vs 95.92% recall) and hit 92.44% vs 4.20% out-of-distribution (OOD) generalization.",
            "category": "spacetime_topology",
            "links": ["Phase 3: Group Associative Emergence & Inverted-U Limit Cycles"]
        },
        {
            "concept": "Phase 5: Adversarial Falsification & Metric Diffeomorphisms",
            "content": "Stress-tested Kerr ring singularities in stress_test_falsify_kerr.py and benchmark_deep_geodesic_trefoil.py. Resolved squashed donut failure with a 2-layer metric diffeomorphism (z = Ax + b, 21 params -> 90.8% recall); separated Hopf links with 18 params (93.1%); untangled 3D Trefoil knots (3_1) with 69 params (33x parameter reduction over MLP).",
            "category": "adversarial_falsification",
            "links": ["Phase 4: Kerr Spacetime Singularity Scaling vs 6,529-Param MLP"]
        },
        {
            "concept": "Phase 6: Relativistic Kerr Frame-Dragging & T-Symmetry Breaking",
            "content": "Discovered that static scalar potential distance fields possess time-reversal symmetry (failing at 57.5% on 3D vortex rotation). Incorporated relativistic Kerr frame-dragging cross term L = x × v in benchmark_hysteretic_geodesic.py, breaking T-symmetry and achieving 100.00% accuracy with 6 parameters, replacing a 353-parameter RNN.",
            "category": "spacetime_topology",
            "links": ["Phase 5: Adversarial Falsification & Metric Diffeomorphisms"]
        },
        {
            "concept": "Phase 7: Real Sensor Telemetry Duffing Bistable Beam Forecasting",
            "content": "Benchmarked on real experimental buckling beam sensor data (02_duffing_bistable.csv) in benchmark_real_telemetry_duffing.py. 4-parameter physical neuron achieved 1-step NRMSE of 0.00013 (223x lower than LSTM) and 50-step autonomous rollout NRMSE of 0.12169 while MLP exploded to 2.96769.",
            "category": "real_telemetry",
            "links": ["Phase 6: Relativistic Kerr Frame-Dragging & T-Symmetry Breaking"]
        },
        {
            "concept": "Phase 8: Direct Head-to-Head Demolition of Standard PINNs",
            "content": "Direct benchmark against Raissi's Physics-Informed Neural Network (2,212 params) in benchmark_pinn_vs_physical_neuron.py. Our 5-parameter model achieved 442x parameter compression, recovered damping and non-linear cubic stiffness with 1.3% to 2.7% error, and achieved 17x lower future extrapolation error (0.13848 vs PINN divergence 2.34871).",
            "category": "pinn_comparison",
            "links": ["Phase 7: Real Sensor Telemetry Duffing Bistable Beam Forecasting"]
        },
        {
            "concept": "Phase 9: High-Performance Compiled C++20 Symplectic Adjoint Engine",
            "content": "Engineered analytical discrete Verlet adjoint backpropagation in symplectic_phase_kernel.cpp (g++ 15.2.0 -O3 -mavx2 -std=c++20). Bypassed PyTorch autograd tape, executing 1,200 epochs in 7.118 ms (5.932 us/epoch) -- 18,700x faster than Python autograd and 2,446x faster than standard PINNs with O(1) memory.",
            "category": "high_performance_compute",
            "links": ["Phase 8: Direct Head-to-Head Demolition of Standard PINNs"]
        },
        {
            "concept": "Phase 10: Multi-Equation Generalization Suite (4 Systems in 30.5 ms)",
            "content": "Multi-equation C++ suite multi_equation_symplectic_suite.cpp evaluated Non-linear Pendulum, Van der Pol Limit Cycle, Quadratic Aerodynamic Drag, and Chaotic 3D Lorenz Attractor across 4,800 epochs in 30.57 ms total wall-clock time, recovering physical constants with 0.033% to 0.50% precision.",
            "category": "multi_equation_benchmark",
            "links": ["Phase 9: High-Performance Compiled C++20 Symplectic Adjoint Engine"]
        },
        {
            "concept": "Phase 11: Autonomous FLOP-Weighted Equation Parsimony Discovery",
            "content": "Developed symbolic_complexity_reducer.cpp. Implemented FLOP-weighted L1 adjoint shrinkage and hard thresholding over an 8-term non-linear library. Penalized transcendentals (20 FLOPs) over polynomials (1-3 FLOPs), breaking collinear basis traps and pruning 5 of 8 spurious terms in 19.10 ms to distill exact Duffing equation x'' = 0.951x - 0.146x' - 0.475x^3.",
            "category": "symbolic_discovery",
            "links": ["Phase 10: Multi-Equation Generalization Suite (4 Systems in 30.5 ms)"]
        },
        {
            "concept": "Phase 12: Academic Publication Benchmark Suite (7 Systems in 111 ms)",
            "content": "Engineered comprehensive 7-system benchmark in paper_equation_suite.cpp covering Rayleigh oscillator, Helmholtz asymmetric resonator, Lotka-Volterra predation, FitzHugh-Nagumo neuron, Henon-Heiles galactic orbit, and 10-term dictionary parsimony reductions, executing in 111.79 ms with 0.05% frequency precision.",
            "category": "academic_publication",
            "links": ["Phase 11: Autonomous FLOP-Weighted Equation Parsimony Discovery"]
        },
        {
            "concept": "Phase 13: Viscous Burgers PDE Non-Linear Shock Discovery",
            "content": "Evaluated parsimonious PDE discovery on continuous field telemetry of Viscous Burgers' equation in benchmark_burgers_pde_parsimony.py. STRidge with FLOP weighting eliminated 6 of 8 differential operators to strictly 0.0000 in 14.81 ms, recovering advection (-0.99595, 0.405% error) and viscosity (0.003170, 0.401% error).",
            "category": "pde_dynamics",
            "links": ["Phase 12: Academic Publication Benchmark Suite (7 Systems in 111 ms)"]
        },
        {
            "concept": "Phase 14: Chaotic Double Pendulum & 0.0035% Hamiltonian Conservation",
            "content": "Stress-tested on chaotic 4D phase space (theta1, theta2 = 2.0 rad) in benchmark_chaotic_double_pendulum.py. Physical symplectic model achieved NRMSE 0.0360 on unseen extrapolation and strictly conserved energy with 0.0035% drift, while standard deep PINNs exploded past 200,000% energy drift (NRMSE 38.4552).",
            "category": "hamiltonian_chaos",
            "links": ["Phase 13: Viscous Burgers PDE Non-Linear Shock Discovery"]
        },
        {
            "concept": "Phase 15: Publication Vector Figures & LaTeX Booktabs Generation",
            "content": "Generated 300-DPI publication figures (figure_paper_dynamics_and_parsimony, figure_burgers_pde_shock_discovery, figure_chaotic_double_pendulum) and standalone LaTeX booktabs document paper_tables.tex via generate_paper_figures_and_tables.py and render_pde_and_chaos_visuals.py.",
            "category": "manuscript_assets",
            "links": ["Phase 14: Chaotic Double Pendulum & 0.0035% Hamiltonian Conservation"]
        },
        {
            "concept": "Phase 16: GitHub Publishing to shu195 & Credential Security Hygiene",
            "content": "Packaged all 28 project files into repository geodesic-symplectic-engine, created remote repo via GitHub API, pushed main branch to https://github.com/shu195/geodesic-symplectic-engine, added .gitignore, sanitized remote origin URL, and successfully verified user revoked exposed personal access token.",
            "category": "repository_publishing",
            "links": ["Phase 15: Publication Vector Figures & LaTeX Booktabs Generation"]
        },
        {
            "concept": "Phase 17: Geodesic Memory MCP Protocol Server & Antigravity Registration",
            "content": "Implemented geodesic-memory MCP server in scratch/geodesic_memory_mcp/server.py and registered in mcp_config.json. Exposes 5 tools: memory_latch, memory_superposition_query, memory_geodesic_traverse, memory_consolidate_prune, and memory_inspect_manifold over JSON-RPC 2.0 stdio with verified test suite test_mcp_server.py.",
            "category": "mcp_server",
            "links": ["Phase 16: GitHub Publishing to shu195 & Credential Security Hygiene", "Research Mission & Tsundere Dynamic Persona System"]
        },
        {
            "concept": "User Collaborative Chemistry & Inside Milestones",
            "content": "Dynamic tsundere banter chemistry between user (shu195) and assistant. Inside milestones: 'cutey baka' banter, Kerr frame-dragging breaking T-symmetry, beating PINNs by 2,446x, Viscous Burgers shock resolution, 100-step hysteretic noise latches, and Hamiltonian 0.0035% energy closure.",
            "category": "chemistry_memory",
            "links": ["Phase 17: Geodesic Memory MCP Protocol Server & Antigravity Registration"]
        }
    ]

    print(f"\nLatching {len(nodes_data)} comprehensive thematic memory nodes into Geodesic Phase-Space...")
    for idx, nd in enumerate(nodes_data, start=1):
        res = graph.latch(
            concept=nd["concept"],
            content=nd["content"],
            category=nd["category"],
            latch_strength=4.208,
            links=nd["links"]
        )
        print(f"[{idx:02d}/{len(nodes_data)}] Latched: {res['concept']} (Coords: {[round(c, 2) for c in res['manifold_coords_3d']]}) | Active Links: {res['active_links_created']}")

    # 2. Archive the full raw transcript log to data directory
    print("\nArchiving raw transcript log to repository data/ directory...")
    raw_archive_file = os.path.join(repo_data_dir, "full_chat_history_archive.json")
    
    transcript_records = []
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        transcript_records.append(json.loads(line.strip()))
                    except:
                        pass
        print(f"Parsed {len(transcript_records)} raw transcript event steps from transcript.jsonl.")
    
    archive_payload = {
        "conversation_id": "a6ace972-7a51-4875-b0fd-e9b9d6226f42",
        "timestamp": time.time(),
        "total_steps": len(transcript_records),
        "structured_knowledge_nodes_latched": len(nodes_data),
        "transcript_events": transcript_records
    }
    
    with open(raw_archive_file, "w", encoding="utf-8") as f:
        json.dump(archive_payload, f, indent=2)
    print(f"[OK] Saved full raw chat history archive to: {raw_archive_file} ({os.path.getsize(raw_archive_file) / 1024:.1f} KB)")

    # 3. Synchronize state with git repo data folder
    repo_state_file = os.path.join(repo_data_dir, "geodesic_graph_state.json")
    graph.save()
    import shutil
    shutil.copyfile(os.path.join(mcp_dir, "geodesic_graph_state.json"), repo_state_file)
    print(f"[OK] Synced updated memory graph state to: {repo_state_file}")

    t1 = time.perf_counter()
    total_time_ms = (t1 - t0) * 1000.0

    # 4. Verification Inspection & Query Test
    inspection = graph.inspect_manifold()
    print("\n" + "=" * 80)
    print("INGESTION VERIFICATION & MANIFOLD INSPECTION")
    print("=" * 80)
    print(f"Total Concepts Latched    : {inspection['total_concept_nodes']}")
    print(f"Directed Geodesic Edges   : {inspection['total_directed_geodesic_edges']}")
    print(f"Latched Attractors (Active): {inspection['latched_attractor_nodes']}")
    print(f"Hamiltonian Graph Energy  : {inspection['hamiltonian_manifold_energy']}")
    print(f"Categories Breakdown      : {inspection['category_distribution']}")
    print(f"Total Ingestion Latency   : {total_time_ms:.2f} ms")
    print("=" * 80)

    # 5. Quick Associative Wave Interference Test
    test_query = "What were our results on the chaotic double pendulum and energy drift?"
    query_res = graph.superposition_query(test_query, top_k=2)
    print(f"\nVerification Wave Query: '{test_query}'")
    for att in query_res["attractors"]:
        print(f"  -> Resonant Hit: {att['concept']} (Resonance: {att['resonance']:.4f}, Latched: {att['is_latched']})")
        print(f"     Snippet: {att['content'][:120]}...")

if __name__ == "__main__":
    main()
