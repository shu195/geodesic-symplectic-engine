import subprocess
import json
import sys

def send_rpc(p, request):
    req_str = json.dumps(request)
    frame = f"Content-Length: {len(req_str.encode('utf-8'))}\r\n\r\n{req_str}"
    p.stdin.write(frame.encode('utf-8'))
    p.stdin.flush()
    
    # Read response
    header = ""
    while True:
        line = p.stdout.readline().decode('utf-8')
        if line.lower().startswith("content-length:"):
            length = int(line.split(":")[1].strip())
            # Read until empty line
            while True:
                sep = p.stdout.readline().decode('utf-8')
                if sep.strip() == "":
                    break
            body = p.stdout.read(length).decode('utf-8')
            return json.loads(body)

def main():
    print("Testing Geodesic Memory MCP Server via stdio JSON-RPC...")
    cmd = [sys.executable, "-u", r"C:\Users\Admin\.gemini\antigravity\scratch\geodesic_memory_mcp\server.py"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # 1. Initialize
        res = send_rpc(p, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        print("[1] Initialize response:", res.get("result", {}).get("serverInfo"))
        
        # 2. Tools List
        res = send_rpc(p, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = res.get("result", {}).get("tools", [])
        print(f"[2] Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
        
        # 3. Latch Node 1 (Kerr Metric)
        res = send_rpc(p, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "memory_latch",
                "arguments": {
                    "concept": "Kerr Frame-Dragging T-Symmetry Breaking",
                    "content": "Using relativistic angular momentum flux L = x × v breaks time-reversal symmetry in 3D coordinate space, tracking vortex rotation with 6 parameters (100% accuracy).",
                    "category": "physics",
                    "latch_strength": 4.208
                }
            }
        })
        text1 = json.loads(res["result"]["content"][0]["text"])
        print("[3] Latch 1:", text1["status"], "| Concept:", text1["concept"])
        
        # 4. Latch Node 2 (Symplectic C++ Kernel) linked to Node 1
        res = send_rpc(p, {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {
                "name": "memory_latch",
                "arguments": {
                    "concept": "Discrete Symplectic C++20 Adjoint Engine",
                    "content": "Bypasses PyTorch autograd graph allocation by unrolling analytical discrete Verlet adjoint equations, executing in 7.1 ms (18,700x faster than autograd).",
                    "category": "performance",
                    "latch_strength": 4.208,
                    "links": ["Kerr Frame-Dragging T-Symmetry Breaking"]
                }
            }
        })
        text2 = json.loads(res["result"]["content"][0]["text"])
        print("[4] Latch 2:", text2["status"], "| Links created:", text2["active_links_created"])
        
        # 5. Superposition Query
        res = send_rpc(p, {
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {
                "name": "memory_superposition_query",
                "arguments": {
                    "query": "How did we break time reversal symmetry and accelerate adjoints?",
                    "top_k": 3
                }
            }
        })
        query_res = json.loads(res["result"]["content"][0]["text"])
        print(f"[5] Query found {query_res['resonant_attractors_found']} attractors:")
        for att in query_res["attractors"]:
            print(f"    - {att['concept']} (Resonance: {att['resonance']}, Latched: {att['is_latched']})")
            
        # 6. Geodesic Traversal
        res = send_rpc(p, {
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {
                "name": "memory_geodesic_traverse",
                "arguments": {
                    "start_concept": "Discrete Symplectic C++20 Adjoint Engine",
                    "max_depth": 2,
                    "direction": "forward"
                }
            }
        })
        trav_res = json.loads(res["result"]["content"][0]["text"])
        print(f"[6] Geodesic Traversal from '{trav_res['origin']}': depth {trav_res['depth_reached']}")
        for step in trav_res["causal_chain"]:
            print(f"    Step {step['step']}: {step['concept']}")
            
        # 7. Manifold Inspection
        res = send_rpc(p, {
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {
                "name": "memory_inspect_manifold",
                "arguments": {}
            }
        })
        insp_res = json.loads(res["result"]["content"][0]["text"])
        print("[7] Manifold Inspection:", insp_res["status"])
        print(f"    Total Nodes: {insp_res['total_concept_nodes']}, Active Edges: {insp_res['total_directed_geodesic_edges']}")
        print(f"    Hamiltonian Energy: {insp_res['hamiltonian_manifold_energy']}")
        
        print("\nALL MCP TESTS PASSED SUCCESSFULLY!")
    finally:
        p.terminate()

if __name__ == "__main__":
    main()
