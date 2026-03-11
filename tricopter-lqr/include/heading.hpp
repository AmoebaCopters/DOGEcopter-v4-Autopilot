#pragma once

#include <Eigen/Dense>

/// Heading-aware position command infrastructure.
///
/// Converts desired translational acceleration (world frame) to
/// commanded roll/pitch angles, accounting for current heading.
///
/// phi_cmd   =  (cos(psi)*ax + sin(psi)*ay) / g
/// theta_cmd = (-sin(psi)*ax + cos(psi)*ay) / g
///
/// For now, desired acceleration is zero (level hover), but the
/// infrastructure exists for a future outer position loop.
struct HeadingAwareCommand {
    /// Compute commanded [phi, theta] from desired world-frame acceleration
    /// and current yaw angle.
    static Eigen::Vector2d computeAttitudeCmd(
        double ax_desired, double ay_desired, double psi);
};
