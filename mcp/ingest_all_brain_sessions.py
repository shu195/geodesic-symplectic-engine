import os
import sys
import json
import time
import glob
import numpy as np

# Path to MCP engine
mcp_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\geodesic_memory_mcp"
sys.path.insert(0, mcp_dir)
from server import GeodesicMemoryGraph, hash_text_to_phase_vector

brain_dir = r"C:\Users\Admin\.gemini\antigravity\brain"
repo_data_dir = r"C:\Users\Admin\.gemini\antigravity\scratch\geodesic-symplectic-engine\data"

def extract_session_summary(session_id: str, transcript_path: str):
    """Parses transcript.jsonl to extract user goals, files touched, and primary topics."""
    user_requests = []
    tool_summaries = []
    created_at = None
    step_count = 0
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                step_count += 1
                try:
                    obj = json.loads(line.strip())
                    if not created_at and "created_at" in obj:
                        created_at = obj["created_at"]
                    
                    # Extract user requests
                    if obj.get("type") == "USER_INPUT":
                        content = obj.get("content", "")
                        if content and len(content.strip()) > 3:
                            # Strip large context summaries
                            if "<CONTEXT_SUMMARY>" in content:
                                parts = content.split("</CONTEXT_SUMMARY>")
                                if len(parts) > 1:
                                    content = parts[-1]
                            clean_text = content.strip().replace("\n", " ")[:200]
                            if clean_text:
                                user_requests.append(clean_text)
                                
                    # Extract tool calls / work summary
                    if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                        for tc in obj["tool_calls"]:
                            fn = tc.get("function", {}).get("name", "")
                            args = tc.get("function", {}).get("arguments", {})
                            if isinstance(args, str):
                                try: args = json.loads(args)
                                except: pass
                            if isinstance(args, dict):
                                summary = args.get("toolSummary") or args.get("Path") or args.get("CommandLine")
                                if summary:
                                    tool_summaries.append(f"{fn}: {str(summary)[:60]}")
                except:
                    continue
    except Exception as e:
        return None
        
    if not user_requests:
        return None
        
    # Title from first representative prompt
    first_prompt = user_requests[0]
    title = f"Session {session_id[:8]}: {first_prompt[:60]}"
    
    # Check if there are artifacts in the folder
    artifacts = []
    sess_dir = os.path.join(brain_dir, session_id)
    for af in glob.glob(os.path.join(sess_dir, "*.md")):
        artifacts.append(os.path.basename(af))
    for af in glob.glob(os.path.join(sess_dir, "*.html")):
        artifacts.append(os.path.basename(af))
        
    content_summary = (
        f"Session ID: {session_id}\n"
        f"Date: {created_at or 'Unknown'}\n"
        f"Total Steps: {step_count}\n"
        f"Primary User Inquiries: {' | '.join(user_requests[:4])}\n"
        f"Artifacts Generated: {', '.join(artifacts[:5]) if artifacts else 'None'}\n"
        f"Actions Taken: {' | '.join(tool_summaries[:4]) if tool_summaries else 'Exploration'}"
    )
    
    return {
        "session_id": session_id,
        "title": title,
        "first_prompt": first_prompt,
        "all_prompts": user_requests,
        "artifacts": artifacts,
        "total_steps": step_count,
        "created_at": created_at,
        "content_summary": content_summary
    }

def main():
    print("=" * 80)
    print("MULTI-SESSION MASS TRANSCRIPT INGESTION ENGINE")
    print("Scanning C:\\Users\\Admin\\.gemini\\antigravity\\brain for all past sessions...")
    print("=" * 80)
    
    t0 = time.perf_counter()
    graph = GeodesicMemoryGraph()
    
    sessions = [d for d in os.listdir(brain_dir) if os.path.isdir(os.path.join(brain_dir, d)) and d != "tempmediaStorage"]
    print(f"Discovered {len(sessions)} total potential session directories.")
    
    parsed_sessions = []
    for s in sessions:
        tpath = os.path.join(brain_dir, s, ".system_generated", "logs", "transcript.jsonl")
        if os.path.exists(tpath) and os.path.getsize(tpath) > 300:
            summary = extract_session_summary(s, tpath)
            if summary:
                parsed_sessions.append(summary)
                
    print(f"Successfully extracted structured trajectory summaries from {len(parsed_sessions)} active sessions!")
    
    # Sort sessions chronologically if date available
    parsed_sessions.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    
    print("\nLatching all session manifolds into Geodesic Phase-Space Memory Graph...")
    latched_count = 0
    
    for s_info in parsed_sessions:
        concept_title = s_info["title"]
        content = s_info["content_summary"]
        sid = s_info["session_id"]
        
        # Link to session if related artifacts or concepts overlap
        links = []
        if any("duffing" in p.lower() or "kerr" in p.lower() or "symplectic" in p.lower() for p in s_info["all_prompts"]):
            links.append("Kerr Metric Spacetime Singularities")
        if any("pinn" in p.lower() or "physics" in p.lower() for p in s_info["all_prompts"]):
            links.append("Discrete Symplectic C++20 Adjoint Engine")
            
        res = graph.latch(
            concept=concept_title,
            content=content,
            category="historical_session",
            latch_strength=4.208,
            links=links
        )
        latched_count += 1
        
    # Save graph state
    graph.save()
    
    # Also save the comprehensive sessions manifest to repo data directory
    manifest_file = os.path.join(repo_data_dir, "all_sessions_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_historical_sessions_indexed": len(parsed_sessions),
            "ingestion_timestamp": time.time(),
            "sessions": parsed_sessions
        }, f, indent=2)
        
    # Copy graph state to repo
    import shutil
    shutil.copyfile(
        os.path.join(mcp_dir, "geodesic_graph_state.json"),
        os.path.join(repo_data_dir, "geodesic_graph_state.json")
    )
    
    t1 = time.perf_counter()
    total_sec = t1 - t0
    
    manifold_info = graph.inspect_manifold()
    
    print("\n" + "=" * 80)
    print("MULTI-SESSION INGESTION REPORT")
    print("=" * 80)
    print(f"Sessions Indexed & Latched    : {latched_count} sessions")
    print(f"Total Graph Concept Nodes     : {manifold_info['total_concept_nodes']}")
    print(f"Total Directed Geodesic Edges : {manifold_info['total_directed_geodesic_edges']}")
    print(f"Hamiltonian Phase-Space Energy: {manifold_info['hamiltonian_manifold_energy']}")
    print(f"Full Manifest Saved To        : {manifest_file} ({os.path.getsize(manifest_file) / 1024:.1f} KB)")
    print(f"Total Ingestion Time          : {total_sec:.2f} seconds ({total_sec / max(1, latched_count) * 1000.0:.2f} ms/session)")
    print("=" * 80)

if __name__ == "__main__":
    main()
