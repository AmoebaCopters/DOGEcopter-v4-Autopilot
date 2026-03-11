#include "heading.hpp"
#include <cmath>

Eigen::Vector2d HeadingAwareCommand::computeAttitudeCmd(
    double ax_desired, double ay_desired, double psi)
{
    const double g = 9.81;
    double cpsi = std::cos(psi);
    double spsi = std::sin(psi);

    double phi_cmd   = ( cpsi * ax_desired + spsi * ay_desired) / g;
    double theta_cmd = (-spsi * ax_desired + cpsi * ay_desired) / g;

    return Eigen::Vector2d(phi_cmd, theta_cmd);
}
