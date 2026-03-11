#include "simulation.hpp"
#include "heading.hpp"
#include "dynamics.hpp"
#include "integrator.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

void Simulation::run(
    const Config& cfg,
    const ControlAllocation& alloc,
    const TrimSolver& trim,
    INDIController& indi,
    const std::string& output_path)
{
    const int N = static_cast<int>(cfg.rotors.size());
    const double dt = cfg.sim.dt;
    const int num_steps = static_cast<int>(cfg.sim.duration / dt);
    const double omega_max_sq = cfg.omega_max * cfg.omega_max;
    const double mg = cfg.mass * 9.81;

    double k_T_avg = 0;
    for (int i = 0; i < N; ++i)
        k_T_avg += cfg.rotors[i].k_T;
    k_T_avg /= N;

    // Altitude PID
    PIDController alt_pid;
    alt_pid.init(cfg.alt_pid);

    // Motor model
    MotorModel motors;
    motors.init(N, trim.u_hover, cfg.motor.tau, omega_max_sq);

    // Priority allocator
    PriorityAllocator priority_alloc;

    // Initial state
    Eigen::VectorXd x = dynamics::buildInitialState(cfg);

    // INDI state
    Eigen::Vector3d omega_prev = Eigen::Vector3d::Zero();
    Eigen::VectorXd u_prev = trim.u_hover;

    // Pre-seed INDI filter with roll/pitch trim angular acceleration only.
    // DO NOT pre-seed yaw: the trim residual yaw torque creates a large
    // angular acceleration (~2.3 rad/s^2) that would cause INDI to demand
    // enormous motor changes through G_inv's large yaw column.
    // Let the filter warm up naturally for yaw.
    Eigen::Matrix3d J_inv = cfg.J.inverse();
    Eigen::Vector3d tau_trim = alloc.B_tau * trim.u_hover;
    Eigen::Vector3d omega_dot_trim = J_inv * tau_trim;
    Eigen::Vector3d preseed_val(omega_dot_trim(0), omega_dot_trim(1), 0.0);
    indi.preseedFilter(preseed_val);

    std::cout << "\n[Sim] Trim angular acceleration: " << omega_dot_trim.transpose() << " rad/s^2\n";
    std::cout << "[Sim] Trim torque residual: " << tau_trim.transpose() << " Nm\n";
    std::cout << "[Sim] Filter pre-seed (RP only): " << preseed_val.transpose() << " rad/s^2\n";

    // Derivative function for integrator
    integrator::DerivFunc deriv_func = dynamics::computeDerivative;

    // Open CSV
    std::ofstream csv(output_path);
    csv << "time,x,y,z,vx,vy,vz,phi_deg,theta_deg,psi_deg,p,q,r,"
        << "u1,u2,u3,T_total,omega_dot_filt_p,omega_dot_filt_q,omega_dot_filt_r\n";

    std::cout << "\n=== Running INDI Simulation ===\n";
    std::cout << "Duration: " << cfg.sim.duration << " s, dt: " << dt << " s\n";
    std::cout << "Initial perturbation: roll=" << cfg.sim.initial_roll_deg
              << " deg, pitch=" << cfg.sim.initial_pitch_deg
              << " deg, yaw=" << cfg.sim.initial_yaw_deg << " deg\n";
    std::cout << "Controller: INDI + Priority Allocator + Altitude PID\n";
    std::cout << "Motor time constant: " << cfg.motor.tau << " s\n\n";

    for (int step = 0; step <= num_steps; ++step) {
        double t = step * dt;

        // --- Extract current attitude ---
        Eigen::Vector3d euler = dynamics::quaternionToEulerZYX(x);
        Eigen::Vector3d omega_body = x.segment<3>(10);

        double phi   = euler(0);
        double theta = euler(1);
        double psi   = euler(2);

        // --- Heading-aware attitude command (level hover) ---
        Eigen::Vector2d att_cmd = HeadingAwareCommand::computeAttitudeCmd(0.0, 0.0, psi);

        // --- Compute motor commands ---
        Eigen::VectorXd u_allocated(N);

        if (step == 0) {
            // First timestep: no valid omega_dot yet, use hover trim
            u_allocated = trim.u_hover;
        } else {
            // --- INDI control ---
            Eigen::Vector3d omega_dot_cmd;
            indi.computeControl(euler, omega_body, omega_prev, dt, att_cmd, omega_dot_cmd);

            // --- Priority allocation (attitude only, no collective) ---
            Eigen::Vector3d delta_omega_dot_des = omega_dot_cmd - indi.omega_dot_filtered;
            Eigen::VectorXd u_attitude = priority_alloc.allocate(
                delta_omega_dot_des, indi.G_inv, u_prev, 0.0, omega_max_sq);

            // --- Altitude PID ---
            double z_current = x(2);
            double z_error = cfg.sim.z_desired - z_current;
            double pid_out = alt_pid.update(z_error, z_current, dt);

            // Tilt-compensated thrust command
            double cos_tilt = std::cos(phi) * std::cos(theta);
            cos_tilt = std::max(cos_tilt, 0.5);
            double T_cmd = (mg + pid_out) / cos_tilt;

            // Post-allocation collective correction: adjust thrust to match T_cmd
            // This is applied AFTER priority allocation to guarantee thrust
            // regardless of how the attitude allocator distributed motors.
            double T_attitude = 0;
            for (int i = 0; i < N; ++i)
                T_attitude += cfg.rotors[i].k_T * u_attitude(i);
            double delta_collective = (T_cmd - T_attitude) / (N * k_T_avg);

            // Apply collective and clamp
            for (int i = 0; i < N; ++i) {
                u_allocated(i) = u_attitude(i) + delta_collective;
                u_allocated(i) = std::clamp(u_allocated(i), 0.0, omega_max_sq);
            }
        }

        // --- Motor model ---
        Eigen::VectorXd u_actual = motors.update(u_allocated, dt);

        // --- Total thrust for logging ---
        double T_total = 0;
        for (int i = 0; i < N; ++i)
            T_total += cfg.rotors[i].k_T * u_actual(i);

        // --- Log to CSV ---
        double phi_deg   = phi   * 180.0 / M_PI;
        double theta_deg = theta * 180.0 / M_PI;
        double psi_deg   = psi   * 180.0 / M_PI;

        csv << std::fixed << std::setprecision(6)
            << t << ","
            << x(0) << "," << x(1) << "," << x(2) << ","
            << x(3) << "," << x(4) << "," << x(5) << ","
            << phi_deg << "," << theta_deg << "," << psi_deg << ","
            << omega_body(0) << "," << omega_body(1) << "," << omega_body(2) << ",";
        for (int i = 0; i < N; ++i)
            csv << std::sqrt(std::max(0.0, u_actual(i))) << ",";
        csv << T_total << ","
            << indi.omega_dot_filtered(0) << ","
            << indi.omega_dot_filtered(1) << ","
            << indi.omega_dot_filtered(2) << "\n";

        // Print samples
        if (step == 0 || step == 1 || step == 2 ||
            step == num_steps - 2 || step == num_steps - 1 || step == num_steps ||
            (step % 1000 == 0)) {
            std::cout << "t=" << std::fixed << std::setprecision(3) << t
                      << "  phi=" << std::setw(8) << std::setprecision(3) << phi_deg
                      << "  theta=" << std::setw(8) << theta_deg
                      << "  psi=" << std::setw(8) << psi_deg
                      << "  r=" << std::setw(8) << std::setprecision(4) << omega_body(2)
                      << "  z=" << std::setw(8) << x(2)
                      << "  T=" << std::setw(8) << std::setprecision(3) << T_total
                      << "\n";
        }

        // --- Disturbance ---
        Eigen::Vector3d ext_torque = Eigen::Vector3d::Zero();
        if (t >= cfg.sim.disturbance_time &&
            t < cfg.sim.disturbance_time + cfg.sim.disturbance_duration) {
            ext_torque = cfg.sim.disturbance_torque;
        }

        // --- Store previous state for next INDI step ---
        omega_prev = omega_body;
        u_prev = u_actual;  // CRITICAL: use actual motor output, not commanded

        // --- Integrate ---
        if (step < num_steps) {
            x = integrator::rk4Step(deriv_func, x, u_actual, cfg, dt, ext_torque);
        }
    }

    csv.close();
    std::cout << "\nSimulation complete. Output: " << output_path << "\n";

    // Print allocator statistics
    priority_alloc.printStatistics();

    std::cout << "===========================\n";
}
