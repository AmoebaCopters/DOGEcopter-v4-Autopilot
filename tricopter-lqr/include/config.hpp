#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

/// Per-rotor configuration
struct RotorConfig {
    std::string name;
    Eigen::Vector3d position;      // position from CG [m], body frame
    Eigen::Vector3d thrust_axis;   // unit vector, thrust direction
    Eigen::Vector3d spin_axis;     // unit vector, spin axis
    int spin_direction;            // +1 CW, -1 CCW
    double k_T;                    // thrust coeff: T = k_T * omega^2
    double k_Q;                    // drag torque coeff: Q = k_Q * omega^2
};

/// INDI controller tuning
struct INDIConfig {
    Eigen::Vector3d Kp_att;    // attitude P gains [roll, pitch, yaw_rate]
    Eigen::Vector3d Kp_rate;   // rate P gains [p, q, r]
    double filter_cutoff_hz;   // angular accel filter cutoff frequency
};

/// Motor dynamics config
struct MotorDynConfig {
    double tau;       // motor time constant (s)
};

/// Allocation config
struct AllocationConfig {
    bool priority_mode;  // enable priority-based allocation
};

/// Altitude PID tuning
struct AltitudePIDConfig {
    double Kp;
    double Ki;
    double Kd;
    double integral_limit;
    double output_limit;
};

/// Simulation parameters
struct SimConfig {
    double dt;
    double duration;
    double initial_roll_deg;
    double initial_pitch_deg;
    double initial_yaw_deg;
    double z_desired;
    double disturbance_time;
    double disturbance_duration;
    Eigen::Vector3d disturbance_torque;  // Nm, body frame
};

/// Top-level configuration
struct Config {
    double mass;                      // kg
    Eigen::Matrix3d J;                // full 3x3 inertia tensor
    std::vector<RotorConfig> rotors;  // variable number of rotors
    INDIConfig indi;
    MotorDynConfig motor;
    AllocationConfig allocation;
    AltitudePIDConfig alt_pid;
    SimConfig sim;
    double omega_max;                 // motor speed limit [rad/s]
};

/// Load configuration from a YAML file. Throws on parse error.
Config loadConfig(const std::string& filepath);
