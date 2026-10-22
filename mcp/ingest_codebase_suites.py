#!/usr/bin/env python3
"""
Ingest Codebase Suites into Geodesic Memory 3.0
Indexes 1,371 Python and C++ source files across 79 project clusters
into Dyadic Multiresolution Code Architecture Nodes.
"""

import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from client import GeodesicMemoryClient

BASE_DIRS = [
    r"C:\Users\Admin\.gemini\antigravity\brain",
    r"C:\Users\Admin\.gemini\antigravity\scratch"
]

def extract_symbols(fp, ext, max_lines=200):
    """Extracts top classes, structs, and functions from source file."""
    symbols = []
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > max_lines:
                    break
                line_str = line.strip()
                if ext == ".py":
                    if line_str.startswith("def ") or line_str.startswith("class "):
                        m = re.match(r"(def|class)\s+([a-zA-Z0-9_]+)", line_str)
                        if m:
                            symbols.append(m.group(2))
                elif ext in [".cpp", ".hpp", ".h"]:
                    if re.match(r"(class|struct)\s+([a-zA-Z0-9_]+)", line_str):
                        m = re.match(r"(class|struct)\s+([a-zA-Z0-9_]+)", line_str)
                        if m:
                            symbols.append(m.group(2))
                    elif re.match(r"(void|int|double|float|bool|auto)\s+([a-zA-Z0-9_]+)\s*\(", line_str):
                        m = re.match(r"(void|int|double|float|bool|auto)\s+([a-zA-Z0-9_]+)\s*\(", line_str)
                        if m:
                            symbols.append(m.group(2))
    except Exception:
        pass
    return list(dict.fromkeys(symbols))[:8]

def main():
    print("="*80)
    print("INGESTING 1,371 PYTHON & C++ CODE FILES INTO GEODESIC MEMORY 3.0")
    print("="*80)

    # 1. Cluster files by project/module
    clusters = defaultdict(list)
    for b in BASE_DIRS:
        if not os.path.exists(b): continue
        for root, dirs, files in os.walk(b):
            dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", "node_modules", ".tempmediaStorage"]]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".py", ".cpp", ".hpp", ".h"]:
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(root, r"C:\Users\Admin\.gemini\antigravity")
                    parts = rel.split(os.sep)
                    if len(parts) >= 2:
                        cluster_key = os.sep.join(parts[:3])
                    else:
                        cluster_key = rel
                    clusters[cluster_key].append((fp, f, ext, os.path.getsize(fp)))

    print(f"Discovered {sum(len(v) for v in clusters.values())} files across {len(clusters)} project clusters.")

    # 2. Ingest clusters via GeodesicMemoryClient
    client = GeodesicMemoryClient()
    try:
        latched_count = 0
        for cluster_name, file_list in clusters.items():
            py_files = [f for f in file_list if f[2] == ".py"]
            cpp_files = [f for f in file_list if f[2] in [".cpp", ".hpp", ".h"]]
            total_kb = sum(f[3] for f in file_list) / 1024.0

            # Determine category and dominant language
            if len(cpp_files) > len(py_files):
                cat = "cpp_engine"
            elif any("test" in f[1].lower() or "benchmark" in f[1].lower() for f in file_list):
                cat = "benchmark_suite"
            elif any("neuron" in cluster_name.lower() or "csrn" in cluster_name.lower() for f in file_list):
                cat = "neural_architecture"
            elif "sandbox" in cluster_name.lower() or "raylib" in cluster_name.lower():
                cat = "sandbox_system"
            else:
                cat = "code_suite"

            concept_title = f"Code Suite: {cluster_name} ({len(file_list)} files, {total_kb:.1f} KB)"

            # Extract symbols from top 6 largest files
            sorted_files = sorted(file_list, key=lambda x: x[3], reverse=True)[:6]
            key_symbols = []
            file_summaries = []
            for fp, fname, ext, fsize in sorted_files:
                syms = extract_symbols(fp, ext)
                if syms:
                    key_symbols.extend(syms)
                    file_summaries.append(f"{fname} ({fsize/1024:.1f} KB): {', '.join(syms)}")
                else:
                    file_summaries.append(f"{fname} ({fsize/1024:.1f} KB)")

            key_symbols = list(dict.fromkeys(key_symbols))[:12]

            # Construct Dyadic Multiresolution Payload
            macro_payload = f"Project {cluster_name}: {len(file_list)} source files ({len(py_files)} Python, {len(cpp_files)} C++), {total_kb:.1f} KB total code. Core symbols: {', '.join(key_symbols[:6])}."
            
            meso_payload = f"Source code suite in {cluster_name}.\nFiles: {len(file_list)} ({len(py_files)} .py, {len(cpp_files)} .cpp/.h, {total_kb:.1f} KB).\nTop Components:\n" + "\n".join(f"  - {fs}" for fs in file_summaries)
            
            all_file_names = [f[1] for f in file_list]
            micro_payload = (
                f"Full File Inventory for {cluster_name} ({len(file_list)} files, {total_kb:.1f} KB):\n"
                f"Files: {', '.join(all_file_names[:40])}"
                + (f" ...and {len(all_file_names)-40} more" if len(all_file_names) > 40 else "")
                + f"\nSymbols Extracted: {', '.join(key_symbols)}\n"
                f"Root Path: C:\\Users\\Admin\\.gemini\\antigravity\\{cluster_name}"
            )

            # Link to related session or concept if applicable
            links = []
            if "brain" in cluster_name:
                # Extract session id
                parts = cluster_name.split(os.sep)
                if len(parts) >= 2:
                    sid_prefix = parts[1][:8]
                    links.append(f"Session {sid_prefix}")
            if "csrn" in cluster_name.lower():
                links.append("Session 6d29eacc")
            if "c_neuron" in cluster_name.lower() or "h_neuron" in cluster_name.lower():
                links.append("Session 7e60f0d8")
            if "geodesic" in cluster_name.lower():
                links.append("Discrete Symplectic C++20 Adjoint Engine")

            res = client.latch(
                concept=concept_title,
                content=micro_payload,
                macro=macro_payload,
                meso=meso_payload,
                micro=micro_payload,
                category=cat,
                links=links
            )
            latched_count += 1
            if latched_count % 15 == 0 or latched_count == len(clusters):
                print(f"  Latched [{latched_count}/{len(clusters)}]: {concept_title[:65]}...")

        # 3. Autonomous Entanglement across the newly added code suites
        print("\nWeaving autonomous wave entanglement across the expanded manifold...")
        entangle_res = client.entangle(min_resonance=0.30, max_edges_per_node=6)
        print("Entanglement Status:", entangle_res["status"])
        print(f"Total Active Geodesic Edges: {entangle_res['active_geodesic_edges_total']}")
        print(f"New Edges Spawned: {entangle_res['new_entangled_edges_spawned']}")

        # 4. Final Manifold Inspection
        insp = client.inspect()
        print("\n" + "="*80)
        print("FINAL GEODESIC MANIFOLD TELEMETRY")
        print("="*80)
        print(f"Total Concept & Code Nodes: {insp['total_concept_nodes']}")
        print(f"Total Directed Geodesic Edges: {insp['total_directed_geodesic_edges']}")
        print(f"Hamiltonian Manifold Energy: {insp['hamiltonian_manifold_energy']}")
        print(f"Category Distribution: {insp['category_distribution']}")
        print("="*80)

    finally:
        client.close()

if __name__ == "__main__":
    main()
