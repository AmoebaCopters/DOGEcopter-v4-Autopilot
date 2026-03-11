#pragma once

#include <Eigen/Dense>

/// 2nd-order Butterworth low-pass filter (discrete, bilinear transform).
///
/// Transfer function:
///   H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
///
/// Coefficients computed from cutoff frequency using pre-warped bilinear transform.
class ButterworthLP2 {
public:
    ButterworthLP2() = default;

    /// Construct and compute coefficients.
    /// cutoff_hz: cutoff frequency in Hz
    /// sample_rate_hz: sampling rate in Hz (typically 1/dt)
    void init(double cutoff_hz, double sample_rate_hz);

    /// Process one sample. Returns filtered output.
    double update(double input);

    /// Reset filter state to zero.
    void reset();

    /// Pre-seed filter state for a known DC value (avoids startup transient).
    void preseed(double dc_value);

    // Coefficients (public for diagnostics)
    double b0 = 0, b1 = 0, b2 = 0;
    double a1 = 0, a2 = 0;

private:
    double x1_ = 0, x2_ = 0;  // previous inputs
    double y1_ = 0, y2_ = 0;  // previous outputs
};

/// 3-axis angular acceleration filter.
/// Wraps 3 independent ButterworthLP2 instances (one per axis).
class AngularAccelFilter {
public:
    AngularAccelFilter() = default;

    /// Initialize all 3 filters with given cutoff and sample rate.
    void init(double cutoff_hz, double sample_rate_hz);

    /// Filter a 3-vector of raw angular acceleration.
    Eigen::Vector3d update(const Eigen::Vector3d& raw_omega_dot);

    /// Reset all 3 filters.
    void reset();

    /// Pre-seed all 3 filters with known DC values.
    void preseed(const Eigen::Vector3d& dc_values);

    /// Access individual filter (for diagnostics).
    const ButterworthLP2& filter(int axis) const { return filters_[axis]; }

private:
    ButterworthLP2 filters_[3];
};
