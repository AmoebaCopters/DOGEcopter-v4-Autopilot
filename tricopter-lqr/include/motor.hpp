#pragma once

#include <Eigen/Dense>
#include <vector>

/// First-order motor dynamics model.
///
/// Each motor has state u_actual (current omega^2) that lags the
/// commanded value with time constant tau:
///   u_actual_new = u_actual + (u_commanded - u_actual) * (dt / tau)
///
/// INDI must use u_actual (not u_commanded) as its baseline to
/// maintain the incremental assumption.
class MotorModel {
public:
    MotorModel() = default;

    /// Initialize N motors at given initial omega^2 values.
    void init(int n_motors, const Eigen::VectorXd& u_initial,
              double tau, double omega_max_sq);

    /// Update all motors given commanded omega^2.
    /// Returns actual omega^2 after dynamics and clamping.
    Eigen::VectorXd update(const Eigen::VectorXd& u_commanded, double dt);

    /// Get current actual motor states.
    const Eigen::VectorXd& actual() const { return u_actual_; }

    double tau() const { return tau_; }

private:
    Eigen::VectorXd u_actual_;
    double tau_ = 0.03;
    double omega_max_sq_ = 0;
    int n_ = 0;
};
