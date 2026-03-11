#include "allocator.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>

Eigen::VectorXd PriorityAllocator::allocate(
    const Eigen::Vector3d& delta_omega_dot_des,
    const Eigen::Matrix3d& G_inv,
    const Eigen::VectorXd& u_prev,
    double delta_u_collective,
    double omega_max_sq)
{
    const int N = static_cast<int>(u_prev.size());
    total_count++;

    // Full delta_u (without priority)
    Eigen::Vector3d delta_u_full = G_inv * delta_omega_dot_des;

    // Candidate without priority
    Eigen::VectorXd u_candidate(N);
    for (int i = 0; i < N; ++i)
        u_candidate(i) = u_prev(i) + delta_u_full(i) + delta_u_collective;

    // Check saturation
    bool saturated = false;
    for (int i = 0; i < N; ++i) {
        if (u_candidate(i) < 0.0 || u_candidate(i) > omega_max_sq) {
            saturated = true;
            break;
        }
    }

    if (!saturated) {
        return u_candidate;
    }

    // --- Priority allocation: separate RP and yaw ---

    // Roll/pitch component only
    Eigen::Vector3d delta_rp(delta_omega_dot_des(0), delta_omega_dot_des(1), 0.0);
    Eigen::Vector3d delta_yaw(0.0, 0.0, delta_omega_dot_des(2));

    Eigen::Vector3d u_rp = G_inv * delta_rp;
    Eigen::Vector3d u_yaw = G_inv * delta_yaw;

    // RP + collective candidate
    Eigen::VectorXd u_candidate_rp(N);
    for (int i = 0; i < N; ++i)
        u_candidate_rp(i) = u_prev(i) + u_rp(i) + delta_u_collective;

    // Check if RP alone saturates
    bool rp_saturated = false;
    for (int i = 0; i < N; ++i) {
        if (u_candidate_rp(i) < 0.0 || u_candidate_rp(i) > omega_max_sq) {
            rp_saturated = true;
            break;
        }
    }

    if (rp_saturated) {
        // Scale RP to fit within bounds
        rp_saturated_count++;
        double scale = 1.0;
        for (int i = 0; i < N; ++i) {
            double u_base = u_prev(i) + delta_u_collective;
            double delta = u_rp(i);
            if (std::abs(delta) < 1e-15) continue;

            double max_scale;
            if (delta > 0) {
                max_scale = (omega_max_sq - u_base) / delta;
            } else {
                max_scale = (0.0 - u_base) / delta;
            }
            max_scale = std::max(0.0, max_scale);
            scale = std::min(scale, max_scale);
        }

        for (int i = 0; i < N; ++i)
            u_candidate_rp(i) = u_prev(i) + scale * u_rp(i) + delta_u_collective;

        // No yaw headroom
        yaw_scaled_count++;
        min_alpha = 0.0;

        // Clamp
        for (int i = 0; i < N; ++i)
            u_candidate_rp(i) = std::clamp(u_candidate_rp(i), 0.0, omega_max_sq);
        return u_candidate_rp;
    }

    // Find maximum alpha for yaw within remaining headroom
    double alpha = 1.0;
    for (int i = 0; i < N; ++i) {
        if (std::abs(u_yaw(i)) < 1e-15) continue;

        double alpha_i;
        if (u_yaw(i) > 0) {
            alpha_i = (omega_max_sq - u_candidate_rp(i)) / u_yaw(i);
        } else {
            alpha_i = (0.0 - u_candidate_rp(i)) / u_yaw(i);
        }
        alpha_i = std::max(0.0, alpha_i);
        alpha = std::min(alpha, alpha_i);
    }
    alpha = std::clamp(alpha, 0.0, 1.0);

    if (alpha < 1.0) {
        yaw_scaled_count++;
        min_alpha = std::min(min_alpha, alpha);
    }

    // Final allocation
    Eigen::VectorXd u_out(N);
    for (int i = 0; i < N; ++i) {
        u_out(i) = u_candidate_rp(i) + alpha * u_yaw(i);
        u_out(i) = std::clamp(u_out(i), 0.0, omega_max_sq);
    }
    return u_out;
}

void PriorityAllocator::printStatistics() const {
    std::cout << "\n=== Yaw Authority Statistics ===\n";
    std::cout << "Yaw authority reduced in " << yaw_scaled_count
              << " out of " << total_count << " timesteps ("
              << (total_count > 0 ? 100.0 * yaw_scaled_count / total_count : 0.0)
              << "%)\n";
    std::cout << "Minimum alpha: " << min_alpha << "\n";
    if (rp_saturated_count > 0) {
        std::cout << "ROLL/PITCH SATURATION in " << rp_saturated_count
                  << " timesteps — vehicle may be uncontrollable!\n";
    }
    std::cout << "================================\n";
}
