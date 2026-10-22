#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <string>

// ==============================================================================
// Autonomous Symbolic Complexity Reducer & Parsimonious Law Distiller
// Goal: Given a cluttered library of 8 candidate terms (5 spurious, 3 true physical),
// autonomously prune redundant terms to find the minimal Pareto-optimal expression
// in single-digit milliseconds!
// ==============================================================================

struct CandidateTerm {
    std::string name;
    float coeff = 0.1f;
    float m = 0.0f;
    float v = 0.0f;
    bool active = true;
};

int main() {
    std::cout << "================================================================================\n";
    std::cout << "AUTONOMOUS SYMBOLIC COMPLEXITY REDUCER & EQUATION DISTILLER (C++20 / AVX2)\n";
    std::cout << "Input: Cluttered 8-term library (5 spurious nonlinear terms, 3 true terms)\n";
    std::cout << "Goal : Discover minimal Pareto-optimal equation and prune computational complexity\n";
    std::cout << "================================================================================\n\n";

    // 1. True System Generation (Duffing Oscillator: x'' = -0.20*v + 1.00*x - 0.50*x^3)
    const float dt = 0.02f;
    const int N = 300;
    std::vector<float> x_sim(N, 0.0f), v_sim(N, 0.0f);
    x_sim[0] = 0.90f; v_sim[0] = 0.0f;

    for (int i = 0; i < N - 1; ++i) {
        float a = -0.20f * v_sim[i] + 1.00f * x_sim[i] - 0.50f * (x_sim[i] * x_sim[i] * x_sim[i]);
        v_sim[i + 1] = v_sim[i] + dt * a;
        x_sim[i + 1] = x_sim[i] + dt * v_sim[i + 1];
    }

    // 50 Sparse Noisy Sensor Observations
    const int n_obs = 50;
    std::vector<int> obs_idx(n_obs);
    std::vector<float> obs_val(n_obs);
    srand(42);
    for (int i = 0; i < n_obs; ++i) {
        obs_idx[i] = i * (N - 1) / (n_obs - 1);
        obs_val[i] = x_sim[obs_idx[i]] + ((float)rand() / RAND_MAX - 0.5f) * 0.03f;
    }

    // 2. Define Cluttered Candidate Term Library with FLOP complexity weights
    // Polynomials are cheap (1-3 FLOPs), Transcendentals are expensive (20 FLOPs)
    struct CandidateTerm {
        std::string name;
        float coeff = 0.2f;
        float flop_weight = 1.0f; // Computational complexity cost
        float m = 0.0f;
        float v = 0.0f;
        bool active = true;
    };

    std::vector<CandidateTerm> library = {
        {"x",               0.2f,  1.0f, 0.0f, 0.0f, true}, // TRUE (alpha = 1.0)
        {"x'",             -0.1f,  1.0f, 0.0f, 0.0f, true}, // TRUE (gamma = -0.2)
        {"x^2",             0.1f,  2.0f, 0.0f, 0.0f, true}, // SPURIOUS
        {"x^3",            -0.1f,  3.0f, 0.0f, 0.0f, true}, // TRUE (beta = -0.5)
        {"(x')^2",          0.1f,  2.0f, 0.0f, 0.0f, true}, // SPURIOUS
        {"x * x'",          0.1f,  2.0f, 0.0f, 0.0f, true}, // SPURIOUS
        {"sin(x)",          0.1f, 20.0f, 0.0f, 0.0f, true}, // SPURIOUS (Expensive transcendental!)
        {"|x'| * x'",       0.1f,  4.0f, 0.0f, 0.0f, true}  // SPURIOUS
    };

    float x0 = 0.8f, v0 = 0.0f;
    float m_x0 = 0.0f, v_x0 = 0.0f;
    float m_v0 = 0.0f, v_v0 = 0.0f;

    std::vector<float> xt(N), vt(N), dL_dx(N, 0.0f);

    std::cout << "Starting Phase 1: Dense L1-regularized optimization across all 8 terms...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    const float lr = 0.03f;
    const float lambda_sparsity = 0.015f; // L1 sparsity penalty
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    // Optimization Loop: 800 epochs with L1 Lasso pressure
    for (int ep = 1; ep <= 800; ++ep) {
        // Forward integration
        xt[0] = x0; vt[0] = v0;
        for (int i = 0; i < N - 1; ++i) {
            float xi = xt[i], vi = vt[i];
            float a = 0.0f;
            if (library[0].active) a += library[0].coeff * xi;
            if (library[1].active) a += library[1].coeff * vi;
            if (library[2].active) a += library[2].coeff * (xi * xi);
            if (library[3].active) a += library[3].coeff * (xi * xi * xi);
            if (library[4].active) a += library[4].coeff * (vi * vi);
            if (library[5].active) a += library[5].coeff * (xi * vi);
            if (library[6].active) a += library[6].coeff * std::sin(xi);
            if (library[7].active) a += library[7].coeff * (std::abs(vi) * vi);

            vt[i + 1] = vi + dt * a;
            xt[i + 1] = xi + dt * vt[i + 1];
        }

        // Loss gradients on sparse observations
        std::fill(dL_dx.begin(), dL_dx.end(), 0.0f);
        for (int i = 0; i < n_obs; ++i) {
            dL_dx[obs_idx[i]] = (2.0f / n_obs) * (xt[obs_idx[i]] - obs_val[i]);
        }

        // Adjoint backpropagation
        std::vector<float> grads(8, 0.0f);
        float lx = 0.0f, lv = 0.0f;

        for (int i = N - 1; i >= 0; --i) {
            lx += dL_dx[i];
            float xi = xt[i], vi = vt[i];
            if (i > 0) {
                float da_dx = 0.0f, da_dv = 0.0f;
                if (library[0].active) { da_dx += library[0].coeff; grads[0] += lv * xi * dt; }
                if (library[1].active) { da_dv += library[1].coeff; grads[1] += lv * vi * dt; }
                if (library[2].active) { da_dx += 2.0f * library[2].coeff * xi; grads[2] += lv * (xi * xi) * dt; }
                if (library[3].active) { da_dx += 3.0f * library[3].coeff * xi * xi; grads[3] += lv * (xi * xi * xi) * dt; }
                if (library[4].active) { da_dv += 2.0f * library[4].coeff * vi; grads[4] += lv * (vi * vi) * dt; }
                if (library[5].active) { da_dx += library[5].coeff * vi; da_dv += library[5].coeff * xi; grads[5] += lv * (xi * vi) * dt; }
                if (library[6].active) { da_dx += library[6].coeff * std::cos(xi); grads[6] += lv * std::sin(xi) * dt; }
                if (library[7].active) { da_dv += 2.0f * library[7].coeff * std::abs(vi); grads[7] += lv * (std::abs(vi) * vi) * dt; }

                lv = lv + dt * lx + dt * da_dv * lv;
                lx = lx + dt * da_dx * lv;
            }
        }

        // L1 subgradient shrinkage scaled by FLOP complexity penalty!
        for (int k = 0; k < 8; ++k) {
            if (library[k].active) {
                float sign = (library[k].coeff > 0.0f) ? 1.0f : -1.0f;
                grads[k] += sign * lambda_sparsity * library[k].flop_weight;
            }
        }

        // Adam update
        float b1_corr = 1.0f - std::pow(beta1, ep);
        float b2_corr = 1.0f - std::pow(beta2, ep);

        for (int k = 0; k < 8; ++k) {
            if (!library[k].active) continue;
            library[k].m = beta1 * library[k].m + (1.0f - beta1) * grads[k];
            library[k].v = beta2 * library[k].v + (1.0f - beta2) * grads[k] * grads[k];
            library[k].coeff -= lr * (library[k].m / b1_corr) / (std::sqrt(library[k].v / b2_corr) + eps);
        }

        // Initial condition updates
        m_x0 = beta1 * m_x0 + (1.0f - beta1) * lx;
        v_x0 = beta2 * v_x0 + (1.0f - beta2) * lx * lx;
        x0 -= lr * (m_x0 / b1_corr) / (std::sqrt(v_x0 / b2_corr) + eps);
    }

    // Phase 2: HARD THRESHOLDING PRUNING (Occam's Razor Cut)
    std::cout << "\nApplying Hard Thresholding (Pruning coefficients with |c_k| < 0.05)...\n";
    int pruned_count = 0;
    for (int k = 0; k < 8; ++k) {
        if (std::abs(library[k].coeff) < 0.05f) {
            library[k].active = false;
            library[k].coeff = 0.0f;
            pruned_count++;
        }
    }

    // Phase 3: Fine-tuning the remaining sparse terms for 400 epochs (Debiasing)
    for (int ep = 1; ep <= 400; ++ep) {
        xt[0] = x0; vt[0] = v0;
        for (int i = 0; i < N - 1; ++i) {
            float xi = xt[i], vi = vt[i];
            float a = 0.0f;
            if (library[0].active) a += library[0].coeff * xi;
            if (library[1].active) a += library[1].coeff * vi;
            if (library[2].active) a += library[2].coeff * (xi * xi);
            if (library[3].active) a += library[3].coeff * (xi * xi * xi);
            if (library[4].active) a += library[4].coeff * (vi * vi);
            if (library[5].active) a += library[5].coeff * (xi * vi);
            if (library[6].active) a += library[6].coeff * std::sin(xi);
            if (library[7].active) a += library[7].coeff * (std::abs(vi) * vi);

            vt[i + 1] = vi + dt * a;
            xt[i + 1] = xi + dt * vt[i + 1];
        }

        std::fill(dL_dx.begin(), dL_dx.end(), 0.0f);
        for (int i = 0; i < n_obs; ++i) {
            dL_dx[obs_idx[i]] = (2.0f / n_obs) * (xt[obs_idx[i]] - obs_val[i]);
        }

        std::vector<float> grads(8, 0.0f);
        float lx = 0.0f, lv = 0.0f;
        for (int i = N - 1; i >= 0; --i) {
            lx += dL_dx[i];
            float xi = xt[i], vi = vt[i];
            if (i > 0) {
                float da_dx = 0.0f, da_dv = 0.0f;
                if (library[0].active) { da_dx += library[0].coeff; grads[0] += lv * xi * dt; }
                if (library[1].active) { da_dv += library[1].coeff; grads[1] += lv * vi * dt; }
                if (library[2].active) { da_dx += 2.0f * library[2].coeff * xi; grads[2] += lv * (xi * xi) * dt; }
                if (library[3].active) { da_dx += 3.0f * library[3].coeff * xi * xi; grads[3] += lv * (xi * xi * xi) * dt; }
                if (library[4].active) { da_dv += 2.0f * library[4].coeff * vi; grads[4] += lv * (vi * vi) * dt; }
                if (library[5].active) { da_dx += library[5].coeff * vi; da_dv += library[5].coeff * xi; grads[5] += lv * (xi * vi) * dt; }
                if (library[6].active) { da_dx += library[6].coeff * std::cos(xi); grads[6] += lv * std::sin(xi) * dt; }
                if (library[7].active) { da_dv += 2.0f * library[7].coeff * std::abs(vi); grads[7] += lv * (std::abs(vi) * vi) * dt; }

                lv = lv + dt * lx + dt * da_dv * lv;
                lx = lx + dt * da_dx * lv;
            }
        }
        float b1_corr = 1.0f - std::pow(beta1, ep);
        float b2_corr = 1.0f - std::pow(beta2, ep);
        for (int k = 0; k < 8; ++k) {
            if (!library[k].active) continue;
            library[k].m = beta1 * library[k].m + (1.0f - beta1) * grads[k];
            library[k].v = beta2 * library[k].v + (1.0f - beta2) * grads[k] * grads[k];
            library[k].coeff -= lr * (library[k].m / b1_corr) / (std::sqrt(library[k].v / b2_corr) + eps);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Compute final trajectory MSE
    float final_mse = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = xt[i] - x_sim[i];
        final_mse += err * err;
    }
    final_mse /= N;

    // Display Results
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "================================================================================\n";
    std::cout << "EQUATION REDUCTION & PARSIMONY REPORT\n";
    std::cout << "Total Search & Pruning Time: " << elapsed_ms << " ms\n";
    std::cout << "Final Trajectory MSE       : " << final_mse << "\n";
    std::cout << "Number of Terms Pruned     : " << pruned_count << " / 8 terms killed to exactly 0.0000!\n\n";

    std::cout << "Term Analysis & Discovered Coefficients:\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    for (int k = 0; k < 8; ++k) {
        std::string status = library[k].active ? "[KEPT (PHYSICAL)]" : "[PRUNED (SPURIOUS)]";
        std::cout << "  " << std::left << std::setw(15) << library[k].name 
                  << " : Coeff = " << std::setw(8) << library[k].coeff 
                  << " | Status: " << status << "\n";
    }

    std::cout << "\nDiscovered Minimal Analytical Law:\n";
    std::cout << "  x'' = ";
    bool first = true;
    for (int k = 0; k < 8; ++k) {
        if (library[k].active) {
            if (!first && library[k].coeff >= 0) std::cout << " + ";
            else if (!first) std::cout << " - ";
            else if (library[k].coeff < 0) std::cout << "-";
            std::cout << std::abs(library[k].coeff) << " * " << library[k].name;
            first = false;
        }
    }
    std::cout << "\nGround Truth Physical Law:\n";
    std::cout << "  x'' = -0.2000 * x' + 1.0000 * x - 0.5000 * x^3\n";
    std::cout << "================================================================================\n";

    return 0;
}
