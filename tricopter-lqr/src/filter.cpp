#include "filter.hpp"
#include <cmath>

void ButterworthLP2::init(double cutoff_hz, double sample_rate_hz) {
    // Pre-warped analog cutoff frequency (bilinear transform)
    double gamma = std::tan(M_PI * cutoff_hz / sample_rate_hz);
    double g2 = gamma * gamma;
    double sqrt2 = std::sqrt(2.0);

    // Denominator: gamma^2 + sqrt(2)*gamma + 1
    double D = g2 + sqrt2 * gamma + 1.0;

    b0 = g2 / D;
    b1 = 2.0 * g2 / D;
    b2 = g2 / D;
    a1 = 2.0 * (g2 - 1.0) / D;
    a2 = (g2 - sqrt2 * gamma + 1.0) / D;

    reset();
}

double ButterworthLP2::update(double input) {
    double output = b0 * input + b1 * x1_ + b2 * x2_ - a1 * y1_ - a2 * y2_;

    // Shift state
    x2_ = x1_;
    x1_ = input;
    y2_ = y1_;
    y1_ = output;

    return output;
}

void ButterworthLP2::reset() {
    x1_ = x2_ = 0.0;
    y1_ = y2_ = 0.0;
}

void ButterworthLP2::preseed(double dc_value) {
    x1_ = x2_ = dc_value;
    y1_ = y2_ = dc_value;
}

// --- AngularAccelFilter ---

void AngularAccelFilter::init(double cutoff_hz, double sample_rate_hz) {
    for (int i = 0; i < 3; ++i)
        filters_[i].init(cutoff_hz, sample_rate_hz);
}

Eigen::Vector3d AngularAccelFilter::update(const Eigen::Vector3d& raw_omega_dot) {
    Eigen::Vector3d filtered;
    for (int i = 0; i < 3; ++i)
        filtered(i) = filters_[i].update(raw_omega_dot(i));
    return filtered;
}

void AngularAccelFilter::reset() {
    for (int i = 0; i < 3; ++i)
        filters_[i].reset();
}

void AngularAccelFilter::preseed(const Eigen::Vector3d& dc_values) {
    for (int i = 0; i < 3; ++i)
        filters_[i].preseed(dc_values(i));
}
