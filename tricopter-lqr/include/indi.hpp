#pragma once

#include "config.hpp"
#include "control_allocation.hpp"
#include "filter.hpp"
#include <Eigen/Dense>

/// INDI (Incremental Nonlinear Dynamic Inversion) attitude controller.
///
/// 3 cascaded loops:
///   1. Attitude: omega_cmd = Kp_att * (att_desired - att_current)
///      (yaw: omega_cmd_r = -Kp_yaw * r)
///   2. Rate: omega_dot_cmd = Kp_rate * (omega_cmd - omega)
///   3. INDI: delta_u = G_inv * (omega_dot_cmd - omega_dot_filtered)
///
/// G = J_inv * B_tau is the ONLY model knowledge needed.
struct INDIController {
    Eigen::Matrix3d G;          // control effectiveness: J_inv * B_tau
    Eigen::Matrix3d G_inv;      // inverse of G
    double G_cond;              // condition number of G

    Eigen::Vector3d Kp_att;     // attitude proportional gains
    Eigen::Vector3d Kp_rate;    // rate proportional gains

    AngularAccelFilter filter;  // angular acceleration low-pass filter
    Eigen::Vector3d omega_dot_filtered;  // last filtered angular acceleration

    /// Build G matrix from config and allocation. Compute G_inv.
    /// Returns true if G is invertible.
    bool build(const Config& cfg, const ControlAllocation& alloc);

    /// Compute INDI control increment.
    ///   euler:      current [phi, theta, psi]
    ///   omega:      current angular velocity (body frame)
    ///   omega_prev: angular velocity from previous timestep
    ///   dt:         timestep
    ///   att_desired: desired [phi, theta] (from heading-aware command)
    /// Returns delta_u (3-vector of omega^2 increments for attitude control).
    /// Also returns omega_dot_cmd through reference parameter.
    Eigen::Vector3d computeControl(
        const Eigen::Vector3d& euler,
        const Eigen::Vector3d& omega,
        const Eigen::Vector3d& omega_prev,
        double dt,
        const Eigen::Vector2d& att_desired,
        Eigen::Vector3d& omega_dot_cmd_out);

    /// Pre-seed the angular acceleration filter with known DC value.
    void preseedFilter(const Eigen::Vector3d& omega_dot_trim);

    /// Print diagnostics: G, G_inv, condition number.
    void printDiagnostics() const;
};
