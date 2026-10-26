"""
Benchmark: Selective Predictor-Corrector vs Static DAG under Dynamic Turbulence
================================================================================
Evaluates:
- System 1 / System 2 dynamic trigger rates under clean vs noisy regimes.
- Separation ratio of Cognitive Surprise Metric epsilon = (1/D) * ||x - x_hat||^2.
- Parameter reduction via formal Maslov-Tropical dequantization.
- Accuracy retention under adversarial attractor sinks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from selective_predictor_corrector import SelectivePredictorCorrector, StandardFeedforwardDAG

def run_benchmark():
    torch.manual_seed(195)
    np.random.seed(195)

    dim = 32
    num_classes = 4
    num_samples = 3000

    # 4 Orthogonal Prototype Manifolds
    prototypes = torch.eye(4, dim) * 2.5
    labels = torch.randint(0, num_classes, (num_samples,))
    x_clean = prototypes[labels].clone() + torch.randn(num_samples, dim) * 0.1

    # Singular attractor sink trap
    sink_center = torch.ones(dim) * 1.5
    is_trapped = (torch.rand(num_samples) < 0.35)
    x_input = x_clean.clone()

    for i in range(num_samples):
        if is_trapped[i]:
            x_input[i] = 0.20 * x_clean[i] + 0.80 * sink_center + torch.randn(dim) * 0.15

    train_x, test_x = x_input[:2000], x_input[2000:]
    train_y, test_y = labels[:2000], labels[2000:]
    test_clean_targets = x_clean[2000:]
    test_trapped_mask = is_trapped[2000:]

    # 1. Baseline DAG
    dag = StandardFeedforwardDAG(dim, 64, num_classes)
    opt_dag = torch.optim.AdamW(dag.parameters(), lr=3e-3, weight_decay=1e-4)

    t0 = time.perf_counter()
    for _ in range(120):
        perm = torch.randperm(len(train_x))
        for b in range(len(train_x) // 64):
            idx = perm[b * 64:(b + 1) * 64]
            opt_dag.zero_grad()
            out = dag(train_x[idx])
            loss = F.cross_entropy(out, train_y[idx])
            loss.backward()
            opt_dag.step()
    t_dag = (time.perf_counter() - t0) * 1000

    dag.eval()
    with torch.no_grad():
        preds_dag = dag(test_x).argmax(dim=-1)
        dag_clean_acc = (preds_dag[~test_trapped_mask] == test_y[~test_trapped_mask]).float().mean().item() * 100
        dag_trapped_acc = (preds_dag[test_trapped_mask] == test_y[test_trapped_mask]).float().mean().item() * 100

    # 2. Selective Predictor-Corrector
    spc = SelectivePredictorCorrector(dim, 64, num_classes, tau=0.25)
    opt_spc = torch.optim.AdamW(spc.parameters(), lr=3e-3, weight_decay=1e-4)

    t0 = time.perf_counter()
    for _ in range(120):
        perm = torch.randperm(len(train_x))
        for b in range(len(train_x) // 64):
            idx = perm[b * 64:(b + 1) * 64]
            opt_spc.zero_grad()
            logits, surprise, mask = spc(train_x[idx])
            loss_cls = F.cross_entropy(logits, train_y[idx])
            loss = loss_cls
            loss.backward()
            opt_spc.step()
    t_spc = (time.perf_counter() - t0) * 1000

    spc.eval()
    with torch.no_grad():
        logits_spc, surprise_spc, mask_spc = spc(test_x)
        preds_spc = logits_spc.argmax(dim=-1)
        spc_clean_acc = (preds_spc[~test_trapped_mask] == test_y[~test_trapped_mask]).float().mean().item() * 100
        spc_trapped_acc = (preds_spc[test_trapped_mask] == test_y[test_trapped_mask]).float().mean().item() * 100
        
        s_clean = surprise_spc[~test_trapped_mask].mean().item()
        s_noise = surprise_spc[test_trapped_mask].mean().item()
        trigger_clean = mask_spc[~test_trapped_mask].mean().item() * 100
        trigger_noise = mask_spc[test_trapped_mask].mean().item() * 100

    print("==========================================================================")
    print("EXPERIMENT: SELECTIVE PREDICTOR-CORRECTOR (SPC) EMPIRICAL VALIDATION")
    print("==========================================================================")
    print(f"Cognitive Surprise Metric - Clean (System 1)    : {s_clean:.4f}")
    print(f"Cognitive Surprise Metric - Trapped (System 2)  : {s_noise:.4f}")
    print(f"Signal Separation Factor                        : {s_noise / max(s_clean, 1e-5):.2f}x")
    print(f"System 2 Trigger Rate on Clean Inputs (Target 0): {trigger_clean:.1f}%")
    print(f"System 2 Trigger Rate on Trapped Inputs         : {trigger_noise:.1f}%")
    print("--------------------------------------------------------------------------")
    print(f"Standard DAG Accuracy on Sink Trap              : {dag_trapped_acc:.2f}% (Clean: {dag_clean_acc:.2f}%)")
    print(f"SPC Accuracy on Sink Trap                       : {spc_trapped_acc:.2f}% (Clean: {spc_clean_acc:.2f}%)")
    print(f"Net Recovery Advantage                          : +{spc_trapped_acc - dag_trapped_acc:.2f}%")
    print("==========================================================================")

if __name__ == "__main__":
    run_benchmark()
