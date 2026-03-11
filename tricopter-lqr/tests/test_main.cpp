#include "config.hpp"
#include "control_allocation.hpp"
#include "trim.hpp"
#include "indi.hpp"
#include "filter.hpp"
#include "motor.hpp"
#include "allocator.hpp"
#include "heading.hpp"
#include "pid.hpp"
#include "dynamics.hpp"
#include "integrator.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <cmath>
#include <cassert>

// Simple test framework
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "TEST: " << #name << " ... "; \
    try { test_##name(); tests_passed++; std::cout << "PASSED\n"; } \
    catch (const std::exception& e) { tests_failed++; std::cout << "FAILED: " << e.what() << "\n"; }

#define ASSERT_TRUE(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond)

#define ASSERT_NEAR(a, b, tol) \
    if (std::abs((a) - (b)) > (tol)) { \
        throw std::runtime_error( \
            "ASSERT_NEAR failed: " #a " = " + std::to_string(a) + \
            ", " #b " = " + std::to_string(b) + \
            ", diff = " + std::to_string(std::abs((a)-(b)))); \
    }

static Config getTestConfig() {
    return loadConfig("config/tricopter_default.yaml");
}

// ===== 1. B_alloc dimensions =====
void test_balloc_dimensions() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    int N = static_cast<int>(cfg.rotors.size());
    ASSERT_TRUE(alloc.B_alloc.rows() == 6);
    ASSERT_TRUE(alloc.B_alloc.cols() == N);
    ASSERT_TRUE(alloc.B_force.rows() == 3);
    ASSERT_TRUE(alloc.B_force.cols() == N);
    ASSERT_TRUE(alloc.B_tau.rows() == 3);
    ASSERT_TRUE(alloc.B_tau.cols() == N);
}

// ===== 2. B_alloc force values =====
void test_balloc_force_values() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(alloc.B_force(0, i), 0.0, 1e-12);
        ASSERT_NEAR(alloc.B_force(1, i), 0.0, 1e-12);
        ASSERT_NEAR(alloc.B_force(2, i), 1.5e-5, 1e-12);
    }
}

// ===== 3. B_alloc torque values =====
void test_balloc_torque_values() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    // Rotor 0 (front): pos=[0.25,0,0], k_T=1.5e-5, spin_dir=+1, k_Q=2.5e-7
    ASSERT_NEAR(alloc.B_tau(0, 0), 0.0, 1e-12);
    ASSERT_NEAR(alloc.B_tau(1, 0), -0.25 * 1.5e-5, 1e-12);
    ASSERT_NEAR(alloc.B_tau(2, 0), 2.5e-7, 1e-12);

    // Rotor 1 (rear-left): pos=[-0.15,0.20,0], spin_dir=-1
    ASSERT_NEAR(alloc.B_tau(0, 1), 0.20 * 1.5e-5, 1e-12);
    ASSERT_NEAR(alloc.B_tau(1, 1), 0.15 * 1.5e-5, 1e-12);
    ASSERT_NEAR(alloc.B_tau(2, 1), -2.5e-7, 1e-12);
}

// ===== 4. Trim residual torque =====
void test_trim_residual_torque() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);
    TrimSolver trim;
    ASSERT_TRUE(trim.solve(cfg, alloc));

    Eigen::Vector3d residual = alloc.B_tau * trim.u_hover;
    std::cout << "    Trim residual torque norm: " << residual.norm() << "\n";
    ASSERT_TRUE(std::isfinite(residual.norm()));
    ASSERT_TRUE(residual.norm() < 1.0);
}

// ===== 5. Trim thrust balance =====
void test_trim_thrust_balance() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);
    TrimSolver trim;
    ASSERT_TRUE(trim.solve(cfg, alloc));

    double mg = cfg.mass * 9.81;
    ASSERT_NEAR(trim.total_thrust_hover, mg, 0.01);
}

// ===== 6. G matrix inverse =====
void test_G_matrix_inverse() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    INDIController indi;
    ASSERT_TRUE(indi.build(cfg, alloc));

    ASSERT_TRUE(indi.G.rows() == 3);
    ASSERT_TRUE(indi.G.cols() == 3);

    Eigen::Matrix3d check = indi.G_inv * indi.G;
    double err = (check - Eigen::Matrix3d::Identity()).norm();
    std::cout << "    G_inv * G identity error: " << err << "\n";
    ASSERT_TRUE(err < 1e-8);
}

// ===== 7. G condition number =====
void test_G_condition_number() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    INDIController indi;
    ASSERT_TRUE(indi.build(cfg, alloc));

    std::cout << "    G condition number: " << indi.G_cond << "\n";
    ASSERT_TRUE(std::isfinite(indi.G_cond));
    ASSERT_TRUE(indi.G_cond > 0);
    ASSERT_TRUE(indi.G_cond < 10000);
}

// ===== 8. Butterworth filter step response =====
void test_butterworth_filter_step() {
    ButterworthLP2 filt;
    filt.init(30.0, 1000.0);

    double y = 0;
    for (int i = 0; i < 200; ++i)
        y = filt.update(1.0);

    std::cout << "    Filter step response at t=0.2s: " << y << "\n";
    ASSERT_NEAR(y, 1.0, 0.01);
}

// ===== 9. Butterworth filter attenuation =====
void test_butterworth_filter_attenuation() {
    ButterworthLP2 filt;
    filt.init(30.0, 1000.0);

    double fs = 1000.0;
    double fc = 30.0;
    int n_samples = 2000;

    double max_out = 0;
    for (int i = 0; i < n_samples; ++i) {
        double t = i / fs;
        double input = std::sin(2.0 * M_PI * fc * t);
        double output = filt.update(input);
        if (i > 1000)
            max_out = std::max(max_out, std::abs(output));
    }

    std::cout << "    Filter gain at cutoff: " << max_out << " (expected ~0.707)\n";
    ASSERT_TRUE(max_out > 0.5);
    ASSERT_TRUE(max_out < 0.9);
}

// ===== 10. INDI zero input =====
void test_indi_zero_input() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    INDIController indi;
    ASSERT_TRUE(indi.build(cfg, alloc));

    Eigen::Vector3d euler(0, 0, 0);
    Eigen::Vector3d omega(0, 0, 0);
    Eigen::Vector3d omega_prev(0, 0, 0);
    Eigen::Vector2d att_desired(0, 0);
    Eigen::Vector3d omega_dot_cmd;

    Eigen::Vector3d delta_u = indi.computeControl(
        euler, omega, omega_prev, 0.001, att_desired, omega_dot_cmd);

    ASSERT_NEAR(omega_dot_cmd.norm(), 0.0, 1e-6);
    std::cout << "    delta_u norm: " << delta_u.norm() << "\n";
}

// ===== 11. Motor step response =====
void test_motor_step_response() {
    double tau = 0.03;
    double omega_max_sq = 2500.0 * 2500.0;
    double dt = 0.001;

    Eigen::VectorXd u_init(3);
    u_init << 300000, 300000, 300000;

    MotorModel motor;
    motor.init(3, u_init, tau, omega_max_sq);

    double step_size = 10000.0;
    Eigen::VectorXd u_cmd(3);
    u_cmd << 300000 + step_size, 300000 + step_size, 300000 + step_size;

    int steps_tau = static_cast<int>(tau / dt);
    for (int i = 0; i < steps_tau; ++i)
        motor.update(u_cmd, dt);

    double actual_change = motor.actual()(0) - 300000.0;
    double expected_63pct = step_size * (1.0 - std::exp(-1.0));
    std::cout << "    At t=tau: actual=" << actual_change
              << " expected(63%)=" << expected_63pct << "\n";
    ASSERT_NEAR(actual_change, expected_63pct, step_size * 0.05);

    for (int i = 0; i < 2 * steps_tau; ++i)
        motor.update(u_cmd, dt);

    double actual_95 = motor.actual()(0) - 300000.0;
    double expected_95pct = step_size * (1.0 - std::exp(-3.0));
    std::cout << "    At t=3*tau: actual=" << actual_95
              << " expected(95%)=" << expected_95pct << "\n";
    ASSERT_NEAR(actual_95, expected_95pct, step_size * 0.05);
}

// ===== 12. Priority allocator: no saturation =====
void test_priority_no_saturation() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    INDIController indi;
    ASSERT_TRUE(indi.build(cfg, alloc));

    PriorityAllocator pa;
    double omega_max_sq = cfg.omega_max * cfg.omega_max;

    Eigen::Vector3d delta_omega_dot(1.0, 1.0, 0.5);
    Eigen::VectorXd u_prev(3);
    u_prev << 300000, 300000, 300000;
    double delta_coll = 0.0;

    Eigen::VectorXd u_out = pa.allocate(
        delta_omega_dot, indi.G_inv, u_prev, delta_coll, omega_max_sq);

    Eigen::Vector3d delta_u_expected = indi.G_inv * delta_omega_dot;
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(u_out(i), u_prev(i) + delta_u_expected(i), 1e-6);
    }
}

// ===== 13. Priority allocator: yaw sacrifice =====
void test_priority_yaw_sacrifice() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);

    INDIController indi;
    ASSERT_TRUE(indi.build(cfg, alloc));

    PriorityAllocator pa;
    double omega_max_sq = cfg.omega_max * cfg.omega_max;

    Eigen::VectorXd u_prev(3);
    u_prev << omega_max_sq * 0.9, omega_max_sq * 0.9, omega_max_sq * 0.9;

    Eigen::Vector3d delta_omega_dot(0.0, 0.0, 50.0);
    double delta_coll = 0.0;

    Eigen::VectorXd u_out = pa.allocate(
        delta_omega_dot, indi.G_inv, u_prev, delta_coll, omega_max_sq);

    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(u_out(i) >= -1e-6);
        ASSERT_TRUE(u_out(i) <= omega_max_sq + 1e-6);
    }

    Eigen::Vector3d delta_u_full = indi.G_inv * delta_omega_dot;
    bool was_scaled = false;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(u_out(i) - (u_prev(i) + delta_u_full(i))) > 1e-6)
            was_scaled = true;
    }
    ASSERT_TRUE(was_scaled);
    std::cout << "    Yaw scaled count: " << pa.yaw_scaled_count << "\n";
}

// ===== 14. Quaternion normalisation =====
void test_quaternion_normalisation() {
    Config cfg = getTestConfig();
    ControlAllocation alloc;
    alloc.build(cfg);
    TrimSolver trim;
    trim.solve(cfg, alloc);

    Eigen::VectorXd x = dynamics::buildInitialState(cfg);

    integrator::DerivFunc f = dynamics::computeDerivative;
    for (int i = 0; i < 100; ++i) {
        x = integrator::rk4Step(f, x, trim.u_hover, cfg, cfg.sim.dt);
    }

    double qnorm = std::sqrt(x(6)*x(6) + x(7)*x(7) + x(8)*x(8) + x(9)*x(9));
    ASSERT_NEAR(qnorm, 1.0, 1e-10);
}

// ===== 15. PID step response =====
void test_pid_step_response() {
    AltitudePIDConfig pid_cfg;
    pid_cfg.Kp = 5.0;
    pid_cfg.Ki = 1.0;
    pid_cfg.Kd = 3.0;
    pid_cfg.integral_limit = 5.0;
    pid_cfg.output_limit = 10.0;

    PIDController pid;
    pid.init(pid_cfg);

    double z = 0.0;
    double v = 0.0;
    double z_des = 1.0;
    double dt = 0.001;
    double m = 1.5;

    for (int i = 0; i < 10000; ++i) {
        double error = z_des - z;
        double u = pid.update(error, z, dt);
        double a = u / m;
        v += a * dt;
        z += v * dt;
    }

    ASSERT_NEAR(z, z_des, 0.1);
}

// ===== 16. Heading-aware command =====
void test_heading_aware_cmd() {
    Eigen::Vector2d cmd0 = HeadingAwareCommand::computeAttitudeCmd(1.0, 0.0, 0.0);
    ASSERT_NEAR(cmd0(0),  1.0 / 9.81, 1e-6);
    ASSERT_NEAR(cmd0(1),  0.0, 1e-6);

    Eigen::Vector2d cmd90 = HeadingAwareCommand::computeAttitudeCmd(1.0, 0.0, M_PI / 2.0);
    ASSERT_NEAR(cmd90(0), 0.0, 1e-6);
    ASSERT_NEAR(cmd90(1), -1.0 / 9.81, 1e-6);

    Eigen::Vector2d cmd_hover = HeadingAwareCommand::computeAttitudeCmd(0.0, 0.0, 1.23);
    ASSERT_NEAR(cmd_hover(0), 0.0, 1e-15);
    ASSERT_NEAR(cmd_hover(1), 0.0, 1e-15);
}

int main() {
    std::cout << "\n=== Running INDI Unit Tests ===\n\n";

    TEST(balloc_dimensions);
    TEST(balloc_force_values);
    TEST(balloc_torque_values);
    TEST(trim_residual_torque);
    TEST(trim_thrust_balance);
    TEST(G_matrix_inverse);
    TEST(G_condition_number);
    TEST(butterworth_filter_step);
    TEST(butterworth_filter_attenuation);
    TEST(indi_zero_input);
    TEST(motor_step_response);
    TEST(priority_no_saturation);
    TEST(priority_yaw_sacrifice);
    TEST(quaternion_normalisation);
    TEST(pid_step_response);
    TEST(heading_aware_cmd);

    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    std::cout << "====================\n";

    return tests_failed > 0 ? 1 : 0;
}
