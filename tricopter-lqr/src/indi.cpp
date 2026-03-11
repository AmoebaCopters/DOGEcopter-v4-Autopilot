#include "indi.hpp"
#include <iostream>
#include <Eigen/SVD>
#include <cmath>

bool INDIController::build(const Config& cfg, const ControlAllocation& alloc) {
    // G = J_inv * B_tau (3x3 for tricopter)
    Eigen::Matrix3d J_inv = cfg.J.inverse();
    Eigen::Matrix3d B_tau_3x3 = alloc.B_tau;
    G = J_inv * B_tau_3x3;

    // Compute condition number via SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(G);
    auto sv = svd.singularValues();
    G_cond = (sv(2) > 1e-15) ? sv(0) / sv(2) : 1e15;

    // Invert G
    double det = G.determinant();
    if (std::abs(det) < 1e-20) {
        std::cerr << "[INDI] G is singular (det=" << det << "), cannot invert!\n";
        return false;
    }
    G_inv = G.inverse();

    // Store gains
    Kp_att = cfg.indi.Kp_att;
    Kp_rate = cfg.indi.Kp_rate;

    // Initialize filter
    double sample_rate = 1.0 / cfg.sim.dt;
    filter.init(cfg.indi.filter_cutoff_hz, sample_rate);
    omega_dot_filtered.setZero();

    printDiagnostics();
    return true;
}

Eigen::Vector3d INDIController::computeControl(
    const Eigen::Vector3d& euler,
    const Eigen::Vector3d& omega,
    const Eigen::Vector3d& omega_prev,
    double dt,
    const Eigen::Vector2d& att_desired,
    Eigen::Vector3d& omega_dot_cmd_out)
{
    // --- Step 1: Raw angular acceleration estimate ---
    Eigen::Vector3d omega_dot_raw = (omega - omega_prev) / dt;

    // --- Step 2: Filter angular acceleration ---
    omega_dot_filtered = filter.update(omega_dot_raw);

    // --- Step 3: Attitude loop (desired angular rates) ---
    double phi = euler(0);
    double theta = euler(1);
    double r = omega(2);

    Eigen::Vector3d omega_cmd;
    omega_cmd(0) = Kp_att(0) * (att_desired(0) - phi);     // roll
    omega_cmd(1) = Kp_att(1) * (att_desired(1) - theta);   // pitch
    omega_cmd(2) = -Kp_att(2) * r;                         // yaw rate damping

    // --- Step 4: Rate loop (desired angular acceleration) ---
    Eigen::Vector3d omega_dot_cmd;
    omega_dot_cmd(0) = Kp_rate(0) * (omega_cmd(0) - omega(0));
    omega_dot_cmd(1) = Kp_rate(1) * (omega_cmd(1) - omega(1));
    omega_dot_cmd(2) = Kp_rate(2) * (omega_cmd(2) - omega(2));

    omega_dot_cmd_out = omega_dot_cmd;

    // --- Step 5: INDI (incremental inversion) ---
    Eigen::Vector3d delta_omega_dot = omega_dot_cmd - omega_dot_filtered;
    Eigen::Vector3d delta_u = G_inv * delta_omega_dot;

    return delta_u;
}

void INDIController::preseedFilter(const Eigen::Vector3d& omega_dot_trim) {
    filter.preseed(omega_dot_trim);
    omega_dot_filtered = omega_dot_trim;
}

void INDIController::printDiagnostics() const {
    std::cout << "\n=== INDI Controller Setup ===\n";
    std::cout << "G = J_inv * B_tau (3x3):\n" << G << "\n\n";
    std::cout << "G_inv (3x3):\n" << G_inv << "\n\n";
    std::cout << "G condition number: " << G_cond << "\n";
    std::cout << "Kp_att: " << Kp_att.transpose() << "\n";
    std::cout << "Kp_rate: " << Kp_rate.transpose() << "\n";

    // Verify G_inv * G = I
    Eigen::Matrix3d check = G_inv * G;
    double identity_err = (check - Eigen::Matrix3d::Identity()).norm();
    std::cout << "G_inv * G identity error: " << identity_err << "\n";
    std::cout << "=============================\n\n";
}
