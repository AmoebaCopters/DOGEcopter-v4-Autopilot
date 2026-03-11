#include "config.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <iostream>

static Eigen::Vector3d readVec3(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 3)
        throw std::runtime_error("Expected 3-element sequence");
    return Eigen::Vector3d(
        node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

Config loadConfig(const std::string& filepath) {
    YAML::Node root = YAML::LoadFile(filepath);
    Config cfg;

    // --- Vehicle ---
    auto veh = root["vehicle"];
    cfg.mass = veh["mass"].as<double>();

    auto inertia = veh["inertia"];
    if (inertia.IsSequence() && inertia.size() == 3) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cfg.J(i, j) = inertia[i][j].as<double>();
    } else {
        double Jxx = inertia["Jxx"].as<double>();
        double Jxy = inertia["Jxy"].as<double>();
        double Jxz = inertia["Jxz"].as<double>();
        double Jyy = inertia["Jyy"].as<double>();
        double Jyz = inertia["Jyz"].as<double>();
        double Jzz = inertia["Jzz"].as<double>();
        cfg.J << Jxx, Jxy, Jxz,
                 Jxy, Jyy, Jyz,
                 Jxz, Jyz, Jzz;
    }

    // --- Rotors ---
    for (const auto& rn : root["rotors"]) {
        RotorConfig rc;
        rc.name          = rn["name"].as<std::string>();
        rc.position      = readVec3(rn["position"]);
        rc.thrust_axis   = readVec3(rn["thrust_axis"]).normalized();
        rc.spin_axis     = readVec3(rn["spin_axis"]).normalized();
        rc.spin_direction = rn["spin_direction"].as<int>();
        rc.k_T           = rn["k_T"].as<double>();
        rc.k_Q           = rn["k_Q"].as<double>();
        cfg.rotors.push_back(rc);
    }

    // --- INDI ---
    auto indi = root["indi"];
    cfg.indi.Kp_att = readVec3(indi["Kp_att"]);
    cfg.indi.Kp_rate = readVec3(indi["Kp_rate"]);
    cfg.indi.filter_cutoff_hz = indi["filter_cutoff_hz"].as<double>();

    // --- Motor ---
    auto mot = root["motor"];
    cfg.motor.tau = mot["tau"].as<double>();
    cfg.omega_max = mot["omega_max"].as<double>();

    // --- Allocation ---
    auto alloc_node = root["allocation"];
    cfg.allocation.priority_mode = alloc_node["priority_mode"].as<bool>();

    // --- Altitude PID ---
    auto pid = root["altitude_pid"];
    cfg.alt_pid.Kp             = pid["Kp"].as<double>();
    cfg.alt_pid.Ki             = pid["Ki"].as<double>();
    cfg.alt_pid.Kd             = pid["Kd"].as<double>();
    cfg.alt_pid.integral_limit = pid["integral_limit"].as<double>();
    cfg.alt_pid.output_limit   = pid["output_limit"].as<double>();

    // --- Simulation ---
    auto sim = root["simulation"];
    cfg.sim.dt                    = sim["dt"].as<double>();
    cfg.sim.duration              = sim["duration"].as<double>();
    cfg.sim.initial_roll_deg      = sim["initial_roll_deg"].as<double>();
    cfg.sim.initial_pitch_deg     = sim["initial_pitch_deg"].as<double>();
    cfg.sim.initial_yaw_deg       = sim["initial_yaw_deg"].as<double>();
    cfg.sim.z_desired             = sim["z_desired"].as<double>();
    cfg.sim.disturbance_time      = sim["disturbance_time"].as<double>();
    cfg.sim.disturbance_duration  = sim["disturbance_duration"].as<double>();
    cfg.sim.disturbance_torque    = readVec3(sim["disturbance_torque"]);

    std::cout << "[Config] Loaded " << cfg.rotors.size() << " rotors, mass="
              << cfg.mass << " kg, omega_max=" << cfg.omega_max << " rad/s\n";
    return cfg;
}
