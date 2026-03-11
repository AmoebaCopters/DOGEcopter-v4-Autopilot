#pragma once

#include "config.hpp"
#include "control_allocation.hpp"
#include "trim.hpp"
#include "indi.hpp"
#include "motor.hpp"
#include "allocator.hpp"
#include "pid.hpp"
#include <Eigen/Dense>
#include <string>

/// Closed-loop INDI simulation.
///
/// Each timestep:
///   1. Extract Euler angles & angular rates from state
///   2. Heading-aware command -> desired [phi, theta] (zero for hover)
///   3. INDI computes delta_u from attitude/rate errors + filtered omega_dot
///   4. Altitude PID computes collective thrust command
///   5. Priority allocator applies delta_u + collective with RP priority
///   6. Motor model: u_actual = motorModel.update(u_allocated, dt)
///   7. Dynamics integrates with u_actual
///   8. Store omega_prev = omega, u_prev = u_actual
///   9. Log to CSV
struct Simulation {
    /// Run full simulation. Writes CSV to output_path.
    void run(const Config& cfg,
             const ControlAllocation& alloc,
             const TrimSolver& trim,
             INDIController& indi,
             const std::string& output_path = "sim_output.csv");
};
