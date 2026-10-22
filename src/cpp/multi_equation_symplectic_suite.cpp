#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <string>

// ==============================================================================
// Multi-Equation Symplectic & Dynamical Adjoint Benchmark Suite (C++20 / AVX2)
// Testing 4 Canonical Physical Systems:
// 1. Non-Linear Pendulum (Transcendental Elliptic: theta'' + gamma*theta' + omega^2*sin(theta) = 0)
// 2. Van der Pol Pacemaker (Limit Cycle: x'' - mu*(1 - x^2)*x' + omega^2*x = 0)
// 3. Quadratic Aerodynamic Drag (Non-Analytic Dissipation: x'' + c*|x'|*x' + omega^2*x = 0)
// 4. Chaotic 3D Lorenz Attractor (Coupled Strange Attractor: dx/dt, dy/dt, dz/dt)
// ==============================================================================

struct AdamOptimizer {
    float lr;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    int step = 0;
    std::vector<float> m;
    std::vector<float> v;
    
    AdamOptimizer(int n_params, float learning_rate = 0.02f) 
        : lr(learning_rate), m(n_params, 0.0f), v(n_params, 0.0f) {}
        
    void update(std::vector<float*>& params, const std::vector<float>& grads) {
        step++;
        float b1_corr = 1.0f - std::pow(beta1, step);
        float b2_corr = 1.0f - std::pow(beta2, step);
        
        for (size_t i = 0; i < params.size(); ++i) {
            float g = grads[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            float m_hat = m[i] / b1_corr;
            float v_hat = v[i] / b2_corr;
            *(params[i]) -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
};

// ==============================================================================
// Benchmark 1: Non-Linear Pendulum (Finite Amplitude)
// theta'' = -gamma * v - omega^2 * sin(theta)
// ==============================================================================
void run_pendulum() {
    const float gamma_true = 0.15f;
    const float omega2_true = 4.00f; // omega = 2.0 rad/s
    const float dt = 0.02f;
    const int N_train = 250;
    const int N_test = 250;
    const int N_total = N_train + N_test;

    std::vector<float> theta_sim(N_total, 0.0f), v_sim(N_total, 0.0f);
    theta_sim[0] = 1.80f; // Large finite amplitude (approx 103 degrees)
    v_sim[0] = 0.0f;

    for (int i = 0; i < N_total - 1; ++i) {
        float a = -gamma_true * v_sim[i] - omega2_true * std::sin(theta_sim[i]);
        v_sim[i + 1] = v_sim[i] + dt * a;
        theta_sim[i + 1] = theta_sim[i] + dt * v_sim[i + 1];
    }

    const int n_obs = 50;
    std::vector<int> obs_idx(n_obs);
    std::vector<float> obs_val(n_obs);
    srand(101);
    for (int i = 0; i < n_obs; ++i) {
        obs_idx[i] = i * (N_train - 1) / (n_obs - 1);
        obs_val[i] = theta_sim[obs_idx[i]] + ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
    }

    float gamma = 0.5f, omega2 = 2.0f, th0 = 1.2f, v0 = 0.0f;
    std::vector<float*> params = {&gamma, &omega2, &th0, &v0};
    AdamOptimizer opt(params.size(), 0.03f);

    std::vector<float> th_traj(N_train), v_traj(N_train), dL_dth(N_train, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int ep = 0; ep < 1200; ++ep) {
        th_traj[0] = th0; v_traj[0] = v0;
        for (int i = 0; i < N_train - 1; ++i) {
            float a = -gamma * v_traj[i] - omega2 * std::sin(th_traj[i]);
            v_traj[i + 1] = v_traj[i] + dt * a;
            th_traj[i + 1] = th_traj[i] + dt * v_traj[i + 1];
        }

        std::fill(dL_dth.begin(), dL_dth.end(), 0.0f);
        for (int i = 0; i < n_obs; ++i) {
            dL_dth[obs_idx[i]] = (2.0f / n_obs) * (th_traj[obs_idx[i]] - obs_val[i]);
        }

        float g_gamma = 0.0f, g_omega2 = 0.0f, lth = 0.0f, lv = 0.0f;
        for (int i = N_train - 1; i >= 0; --i) {
            lth += dL_dth[i];
            float thi = th_traj[i], vi = v_traj[i];
            if (i > 0) {
                float da_dth = -omega2 * std::cos(thi);
                float da_dv  = -gamma;
                g_gamma  += lv * (-vi) * dt;
                g_omega2 += lv * (-std::sin(thi)) * dt;
                lv = lv + dt * lth + dt * da_dv * lv;
                lth = lth + dt * da_dth * lv;
            }
        }
        opt.update(params, {g_gamma, g_omega2, lth, lv});
    }
    double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    // Extrapolation on test horizon
    std::vector<float> th_test(N_total), v_test(N_total);
    th_test[0] = th0; v_test[0] = v0;
    for (int i = 0; i < N_total - 1; ++i) {
        float a = -gamma * v_test[i] - omega2 * std::sin(th_test[i]);
        v_test[i + 1] = v_test[i] + dt * a;
        th_test[i + 1] = th_test[i] + dt * v_test[i + 1];
    }
    float sse = 0.0f, mean = 0.0f, var = 0.0f;
    for (int i = N_train; i < N_total; ++i) mean += theta_sim[i];
    mean /= N_test;
    for (int i = N_train; i < N_total; ++i) {
        sse += (th_test[i] - theta_sim[i]) * (th_test[i] - theta_sim[i]);
        var += (theta_sim[i] - mean) * (theta_sim[i] - mean);
    }
    float nrmse = std::sqrt(sse / N_test) / std::sqrt(var / N_test);

    std::cout << "1. Non-Linear Pendulum (sin theta)\n";
    std::cout << "   Wall-Clock Time: " << elapsed_ms << " ms (" << (elapsed_ms/1.2) << " microsec/epoch)\n";
    std::cout << "   Extrap NRMSE   : " << nrmse << "\n";
    std::cout << "   gamma  (Truth: " << gamma_true << ") -> Discovered: " << gamma << " (Err: " << std::abs(gamma - gamma_true)/gamma_true * 100 << "%)\n";
    std::cout << "   omega2 (Truth: " << omega2_true << ") -> Discovered: " << omega2 << " (Err: " << std::abs(omega2 - omega2_true)/omega2_true * 100 << "%)\n";
}

// ==============================================================================
// Benchmark 2: Van der Pol Pacemaker (Limit Cycle)
// x'' = mu * (1 - x^2) * v - omega^2 * x
// ==============================================================================
void run_vanderpol() {
    const float mu_true = 0.80f;
    const float omega2_true = 1.00f;
    const float dt = 0.02f;
    const int N_train = 250, N_test = 250, N_total = N_train + N_test;

    std::vector<float> x_sim(N_total, 0.0f), v_sim(N_total, 0.0f);
    x_sim[0] = 0.50f; v_sim[0] = 0.0f;

    for (int i = 0; i < N_total - 1; ++i) {
        float a = mu_true * (1.0f - x_sim[i] * x_sim[i]) * v_sim[i] - omega2_true * x_sim[i];
        v_sim[i + 1] = v_sim[i] + dt * a;
        x_sim[i + 1] = x_sim[i] + dt * v_sim[i + 1];
    }

    const int n_obs = 50;
    std::vector<int> obs_idx(n_obs);
    std::vector<float> obs_val(n_obs);
    srand(102);
    for (int i = 0; i < n_obs; ++i) {
        obs_idx[i] = i * (N_train - 1) / (n_obs - 1);
        obs_val[i] = x_sim[obs_idx[i]] + ((float)rand() / RAND_MAX - 0.5f) * 0.03f;
    }

    float mu = 0.3f, omega2 = 0.5f, x0 = 0.3f, v0 = 0.0f;
    std::vector<float*> params = {&mu, &omega2, &x0, &v0};
    AdamOptimizer opt(params.size(), 0.02f);

    std::vector<float> x_traj(N_train), v_traj(N_train), dL_dx(N_train, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int ep = 0; ep < 1200; ++ep) {
        x_traj[0] = x0; v_traj[0] = v0;
        for (int i = 0; i < N_train - 1; ++i) {
            float a = mu * (1.0f - x_traj[i] * x_traj[i]) * v_traj[i] - omega2 * x_traj[i];
            v_traj[i + 1] = v_traj[i] + dt * a;
            x_traj[i + 1] = x_traj[i] + dt * v_traj[i + 1];
        }

        std::fill(dL_dx.begin(), dL_dx.end(), 0.0f);
        for (int i = 0; i < n_obs; ++i) {
            dL_dx[obs_idx[i]] = (2.0f / n_obs) * (x_traj[obs_idx[i]] - obs_val[i]);
        }

        float g_mu = 0.0f, g_omega2 = 0.0f, lx = 0.0f, lv = 0.0f;
        for (int i = N_train - 1; i >= 0; --i) {
            lx += dL_dx[i];
            float xi = x_traj[i], vi = v_traj[i];
            if (i > 0) {
                float da_dx = -2.0f * mu * xi * vi - omega2;
                float da_dv = mu * (1.0f - xi * xi);
                g_mu     += lv * (1.0f - xi * xi) * vi * dt;
                g_omega2 += lv * (-xi) * dt;
                lv = lv + dt * lx + dt * da_dv * lv;
                lx = lx + dt * da_dx * lv;
            }
        }
        opt.update(params, {g_mu, g_omega2, lx, lv});
    }
    double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    // Extrapolation
    std::vector<float> x_test(N_total), v_test(N_total);
    x_test[0] = x0; v_test[0] = v0;
    for (int i = 0; i < N_total - 1; ++i) {
        float a = mu * (1.0f - x_test[i] * x_test[i]) * v_test[i] - omega2 * x_test[i];
        v_test[i + 1] = v_test[i] + dt * a;
        x_test[i + 1] = x_test[i] + dt * v_test[i + 1];
    }
    float sse = 0.0f, mean = 0.0f, var = 0.0f;
    for (int i = N_train; i < N_total; ++i) mean += x_sim[i];
    mean /= N_test;
    for (int i = N_train; i < N_total; ++i) {
        sse += (x_test[i] - x_sim[i]) * (x_test[i] - x_sim[i]);
        var += (x_sim[i] - mean) * (x_sim[i] - mean);
    }
    float nrmse = std::sqrt(sse / N_test) / std::sqrt(var / N_test);

    std::cout << "\n2. Van der Pol Pacemaker (Limit Cycle Oscillator)\n";
    std::cout << "   Wall-Clock Time: " << elapsed_ms << " ms (" << (elapsed_ms/1.2) << " microsec/epoch)\n";
    std::cout << "   Extrap NRMSE   : " << nrmse << "\n";
    std::cout << "   mu     (Truth: " << mu_true << ") -> Discovered: " << mu << " (Err: " << std::abs(mu - mu_true)/mu_true * 100 << "%)\n";
    std::cout << "   omega2 (Truth: " << omega2_true << ") -> Discovered: " << omega2 << " (Err: " << std::abs(omega2 - omega2_true)/omega2_true * 100 << "%)\n";
}

// ==============================================================================
// Benchmark 3: Non-Analytic Quadratic Aerodynamic Drag
// x'' = -c * |v| * v - omega^2 * x
// ==============================================================================
void run_quadratic_drag() {
    const float c_true = 0.35f;
    const float omega2_true = 2.50f;
    const float dt = 0.02f;
    const int N_train = 250, N_test = 250, N_total = N_train + N_test;

    std::vector<float> x_sim(N_total, 0.0f), v_sim(N_total, 0.0f);
    x_sim[0] = 1.50f; v_sim[0] = 0.0f;

    for (int i = 0; i < N_total - 1; ++i) {
        float a = -c_true * std::abs(v_sim[i]) * v_sim[i] - omega2_true * x_sim[i];
        v_sim[i + 1] = v_sim[i] + dt * a;
        x_sim[i + 1] = x_sim[i] + dt * v_sim[i + 1];
    }

    const int n_obs = 50;
    std::vector<int> obs_idx(n_obs);
    std::vector<float> obs_val(n_obs);
    srand(103);
    for (int i = 0; i < n_obs; ++i) {
        obs_idx[i] = i * (N_train - 1) / (n_obs - 1);
        obs_val[i] = x_sim[obs_idx[i]] + ((float)rand() / RAND_MAX - 0.5f) * 0.03f;
    }

    float c = 0.1f, omega2 = 1.0f, x0 = 1.0f, v0 = 0.0f;
    std::vector<float*> params = {&c, &omega2, &x0, &v0};
    AdamOptimizer opt(params.size(), 0.02f);

    std::vector<float> x_traj(N_train), v_traj(N_train), dL_dx(N_train, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int ep = 0; ep < 1200; ++ep) {
        x_traj[0] = x0; v_traj[0] = v0;
        for (int i = 0; i < N_train - 1; ++i) {
            float a = -c * std::abs(v_traj[i]) * v_traj[i] - omega2 * x_traj[i];
            v_traj[i + 1] = v_traj[i] + dt * a;
            x_traj[i + 1] = x_traj[i] + dt * v_traj[i + 1];
        }

        std::fill(dL_dx.begin(), dL_dx.end(), 0.0f);
        for (int i = 0; i < n_obs; ++i) {
            dL_dx[obs_idx[i]] = (2.0f / n_obs) * (x_traj[obs_idx[i]] - obs_val[i]);
        }

        float g_c = 0.0f, g_omega2 = 0.0f, lx = 0.0f, lv = 0.0f;
        for (int i = N_train - 1; i >= 0; --i) {
            lx += dL_dx[i];
            float xi = x_traj[i], vi = v_traj[i];
            if (i > 0) {
                float da_dx = -omega2;
                float da_dv = -2.0f * c * std::abs(vi);
                g_c      += lv * (-std::abs(vi) * vi) * dt;
                g_omega2 += lv * (-xi) * dt;
                lv = lv + dt * lx + dt * da_dv * lv;
                lx = lx + dt * da_dx * lv;
            }
        }
        opt.update(params, {g_c, g_omega2, lx, lv});
    }
    double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    // Extrapolation
    std::vector<float> x_test(N_total), v_test(N_total);
    x_test[0] = x0; v_test[0] = v0;
    for (int i = 0; i < N_total - 1; ++i) {
        float a = -c * std::abs(v_test[i]) * v_test[i] - omega2 * x_test[i];
        v_test[i + 1] = v_test[i] + dt * a;
        x_test[i + 1] = x_test[i] + dt * v_test[i + 1];
    }
    float sse = 0.0f, mean = 0.0f, var = 0.0f;
    for (int i = N_train; i < N_total; ++i) mean += x_sim[i];
    mean /= N_test;
    for (int i = N_train; i < N_total; ++i) {
        sse += (x_test[i] - x_sim[i]) * (x_test[i] - x_sim[i]);
        var += (x_sim[i] - mean) * (x_sim[i] - mean);
    }
    float nrmse = std::sqrt(sse / N_test) / std::sqrt(var / N_test);

    std::cout << "\n3. Quadratic Aerodynamic Drag (Non-Analytic c*|v|*v)\n";
    std::cout << "   Wall-Clock Time: " << elapsed_ms << " ms (" << (elapsed_ms/1.2) << " microsec/epoch)\n";
    std::cout << "   Extrap NRMSE   : " << nrmse << "\n";
    std::cout << "   c      (Truth: " << c_true << ") -> Discovered: " << c << " (Err: " << std::abs(c - c_true)/c_true * 100 << "%)\n";
    std::cout << "   omega2 (Truth: " << omega2_true << ") -> Discovered: " << omega2 << " (Err: " << std::abs(omega2 - omega2_true)/omega2_true * 100 << "%)\n";
}

// ==============================================================================
// Benchmark 4: Chaotic 3D Lorenz Strange Attractor (Coupled System)
// dx/dt = sigma*(y - x)
// dy/dt = x*(rho - z) - y
// dz/dt = x*y - beta*z
// ==============================================================================
void run_lorenz() {
    const float sigma_true = 10.0f;
    const float rho_true   = 28.0f;
    const float beta_true  = 8.0f / 3.0f; // 2.6667
    const float dt = 0.01f;
    const int N_train = 300, N_test = 200, N_total = N_train + N_test;

    std::vector<float> xs(N_total), ys(N_total), zs(N_total);
    xs[0] = 1.0f; ys[0] = 1.0f; zs[0] = 1.0f;

    for (int i = 0; i < N_total - 1; ++i) {
        float dx = sigma_true * (ys[i] - xs[i]);
        float dy = xs[i] * (rho_true - zs[i]) - ys[i];
        float dz = xs[i] * ys[i] - beta_true * zs[i];
        xs[i + 1] = xs[i] + dt * dx;
        ys[i + 1] = ys[i] + dt * dy;
        zs[i + 1] = zs[i] + dt * dz;
    }

    const int n_obs = 60;
    std::vector<int> obs_idx(n_obs);
    std::vector<float> obs_x(n_obs), obs_y(n_obs), obs_z(n_obs);
    srand(104);
    for (int i = 0; i < n_obs; ++i) {
        obs_idx[i] = i * (N_train - 1) / (n_obs - 1);
        obs_x[i] = xs[obs_idx[i]] + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        obs_y[i] = ys[obs_idx[i]] + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        obs_z[i] = zs[obs_idx[i]] + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    float sigma = 5.0f, rho = 20.0f, beta = 1.5f;
    float x0 = 0.8f, y0 = 0.8f, z0 = 0.8f;
    std::vector<float*> params = {&sigma, &rho, &beta, &x0, &y0, &z0};
    AdamOptimizer opt(params.size(), 0.05f);

    std::vector<float> xt(N_train), yt(N_train), zt(N_train);
    std::vector<float> dLx(N_train, 0.0f), dLy(N_train, 0.0f), dLz(N_train, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int ep = 0; ep < 1200; ++ep) {
        xt[0] = x0; yt[0] = y0; zt[0] = z0;
        for (int i = 0; i < N_train - 1; ++i) {
            xt[i + 1] = xt[i] + dt * (sigma * (yt[i] - xt[i]));
            yt[i + 1] = yt[i] + dt * (xt[i] * (rho - zt[i]) - yt[i]);
            zt[i + 1] = zt[i] + dt * (xt[i] * yt[i] - beta * zt[i]);
        }

        std::fill(dLx.begin(), dLx.end(), 0.0f);
        std::fill(dLy.begin(), dLy.end(), 0.0f);
        std::fill(dLz.begin(), dLz.end(), 0.0f);
        for (int i = 0; i < n_obs; ++i) {
            int k = obs_idx[i];
            dLx[k] = (2.0f / (3 * n_obs)) * (xt[k] - obs_x[i]);
            dLy[k] = (2.0f / (3 * n_obs)) * (yt[k] - obs_y[i]);
            dLz[k] = (2.0f / (3 * n_obs)) * (zt[k] - obs_z[i]);
        }

        float g_sigma = 0.0f, g_rho = 0.0f, g_beta = 0.0f;
        float lx = 0.0f, ly = 0.0f, lz = 0.0f;

        for (int i = N_train - 1; i >= 0; --i) {
            lx += dLx[i]; ly += dLy[i]; lz += dLz[i];
            float xi = xt[i], yi = yt[i], zi = zt[i];
            if (i > 0) {
                // Parameter gradients
                g_sigma += lx * (yi - xi) * dt;
                g_rho   += ly * (xi) * dt;
                g_beta  += lz * (-zi) * dt;

                // 3D Adjoint Jacobian backpropagation: dlambda/dt = -J^T * lambda
                float nlx = lx + dt * (-sigma * lx + (rho - zi) * ly + yi * lz);
                float nly = ly + dt * (sigma * lx - ly + xi * lz);
                float nlz = lz + dt * (-xi * ly - beta * lz);
                lx = nlx; ly = nly; lz = nlz;
            }
        }
        opt.update(params, {g_sigma, g_rho, g_beta, lx, ly, lz});
    }
    double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    std::cout << "\n4. Chaotic 3D Lorenz Strange Attractor\n";
    std::cout << "   Wall-Clock Time: " << elapsed_ms << " ms (" << (elapsed_ms/1.2) << " microsec/epoch)\n";
    std::cout << "   sigma (Truth: " << sigma_true << ") -> Discovered: " << sigma << " (Err: " << std::abs(sigma - sigma_true)/sigma_true * 100 << "%)\n";
    std::cout << "   rho   (Truth: " << rho_true   << ") -> Discovered: " << rho   << " (Err: " << std::abs(rho - rho_true)/rho_true * 100 << "%)\n";
    std::cout << "   beta  (Truth: " << beta_true  << ") -> Discovered: " << beta  << " (Err: " << std::abs(beta - beta_true)/beta_true * 100 << "%)\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "================================================================================\n";
    std::cout << "MULTI-EQUATION SYMPLECTIC & DYNAMICAL ADJOINT BENCHMARK SUITE\n";
    std::cout << "Executing 1,200 training epochs across 4 distinct physical differential equations\n";
    std::cout << "================================================================================\n\n";

    auto total_start = std::chrono::high_resolution_clock::now();

    run_pendulum();
    run_vanderpol();
    run_quadratic_drag();
    run_lorenz();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "\n================================================================================\n";
    std::cout << "TOTAL TIME FOR ALL 4 PHYSICAL SYSTEMS (4 x 1,200 = 4,800 EPOCHS): " << total_ms << " ms\n";
    std::cout << "================================================================================\n";

    return 0;
}
