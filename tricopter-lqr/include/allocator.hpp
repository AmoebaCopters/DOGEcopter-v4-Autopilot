#pragma once

#include <Eigen/Dense>

/// Priority-based control allocator.
///
/// When motors saturate, sacrifices yaw before roll/pitch:
///   1. Decompose delta_omega_dot into roll/pitch and yaw components
///   2. Map each through G_inv to motor commands
///   3. Apply RP + collective fully
///   4. Scale yaw component by alpha in [0,1] to fit within motor limits
///
/// Statistics tracked for diagnostics.
struct PriorityAllocator {
    int yaw_scaled_count = 0;     // timesteps where alpha < 1
    int total_count = 0;          // total timesteps
    double min_alpha = 1.0;       // minimum alpha seen
    int rp_saturated_count = 0;   // timesteps where RP alone saturates

    /// Allocate motor commands with roll/pitch priority.
    ///
    /// delta_omega_dot_des: desired angular acceleration increment (3-vector)
    /// G_inv: inverse control effectiveness matrix
    /// u_prev: previous actual motor commands (N-vector)
    /// delta_u_collective: collective thrust change per motor
    /// omega_max_sq: motor limit (omega_max^2)
    ///
    /// Returns allocated motor commands u (N-vector), already clamped.
    Eigen::VectorXd allocate(
        const Eigen::Vector3d& delta_omega_dot_des,
        const Eigen::Matrix3d& G_inv,
        const Eigen::VectorXd& u_prev,
        double delta_u_collective,
        double omega_max_sq);

    /// Print end-of-simulation statistics.
    void printStatistics() const;
};
