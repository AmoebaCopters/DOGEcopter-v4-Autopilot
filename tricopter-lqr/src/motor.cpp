#include "motor.hpp"
#include <algorithm>

void MotorModel::init(int n_motors, const Eigen::VectorXd& u_initial,
                      double tau, double omega_max_sq) {
    n_ = n_motors;
    u_actual_ = u_initial;
    tau_ = tau;
    omega_max_sq_ = omega_max_sq;
}

Eigen::VectorXd MotorModel::update(const Eigen::VectorXd& u_commanded, double dt) {
    double alpha = dt / tau_;
    for (int i = 0; i < n_; ++i) {
        u_actual_(i) += (u_commanded(i) - u_actual_(i)) * alpha;
        u_actual_(i) = std::clamp(u_actual_(i), 0.0, omega_max_sq_);
    }
    return u_actual_;
}
