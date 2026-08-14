import time

import numpy as np

from mujoco_irb120.robot.controllers.genesis_robot import GenesisRobotController
from mujoco_irb120.robot.controllers.genesis_robot import as_numpy, damped_pseudoinverse, limit_joint_velocity


def velocity_shove(
    robot,
    scene,
    preshove_pos,
    preshove_quat,
    push_direction,
    shove_speed,
    obj             = None,
    camera          = None,
    ramp_up_steps   = 25,
    hold_steps      = 50,
    ramp_down_steps = 25,
    settle_steps    = 100,
    height_kp       = 4.0,
    posture_kp      = 0.75,
    timeout         = 25.0,
    ):
    """
    Execute a guarded Cartesian velocity shove.

    The command pipeline is:
        trapezoid speed
        * workspace ellipsoid scale
        * manipulability scale
        -> damped least-squares joint velocity
        -> joint velocity clamp
    """
    start_time = time.time()

    def step_once():
        scene.step()
        if camera is not None:
            camera.render()


    # TESTING WITHOUT MOTION PLANNING
    # Snap to preshove pose, but first convert cartesian to joint angles
    q_preshove = robot.entity.inverse_kinematics(link=robot.pusher, pos=preshove_pos, quat=preshove_quat)
    robot.entity.set_dofs_position(q_preshove, robot.dofs_idx)
    q_preshove = as_numpy(q_preshove)

    # Briefly hold the preshove pose before making contact.
    for _ in range(50):
        robot.entity.control_dofs_position(q_preshove, robot.dofs_idx)
        step_once()

    # Hold the contact point's vertical position from the start of the shove. Lateral y motion is left unconstrained on purpose.
    contact_ref = robot.link_local_point_world()
    prev_contact_pos = contact_ref.copy()

    shove_steps = ramp_up_steps + hold_steps + ramp_down_steps
    for step in range(shove_steps):
        contact_pos = robot.link_local_point_world()
        jac = robot.get_contact_jacobian()
        jac_pos = jac[:3, :]

        commanded_speed = robot.trapezoidal_plus_scaling_speed(
            step,
            ramp_up_steps,
            hold_steps,
            ramp_down_steps,
            shove_speed,
            contact_pos,
            jac_pos,
        )

        if commanded_speed <= 1e-6:
            break
        elif (time.time() - start_time) > timeout:
            print("Shove timeout reached.")
            break

        # Feedforward shove speed plus feedback to keep y/z near preshove 
        # contact point. For the default +x shove, this gives constant-height 
        # Cartesian command without constraining lateral y.
        target_velocity = commanded_speed * push_direction
        target_velocity[2] += height_kp * (contact_ref[2] - contact_pos[2])

        jac_pinv = damped_pseudoinverse(jac_pos, damping=0.01)
        qdot_task = jac_pinv @ target_velocity

        # The translational task only uses 3 constraints for a 6-DOF arm.
        # This nullspace term asks the unused DOFs to stay near the
        # preshove posture instead of drifting into odd elbow/wrist poses.
        q_current       = as_numpy(robot.entity.get_dofs_position(robot.dofs_idx))
        qdot_posture    = posture_kp * (q_preshove - q_current)
        nullspace       = np.eye(len(robot.dofs_idx)) - jac_pinv @ jac_pos
        qdot_unlimited  = qdot_task + nullspace @ qdot_posture

        qdot, qdot_scale = limit_joint_velocity(qdot_unlimited, robot.joint_velocity_limit)
        predicted_velocity = jac_pos @ qdot

        ## APPLY THE VELOCITY COMMAND AND STEP THE SIMULATION
        robot.entity.control_dofs_velocity(qdot, robot.dofs_idx)
        step_once()

        new_contact_pos = robot.link_local_point_world()
        actual_delta = new_contact_pos - prev_contact_pos
        prev_contact_pos = new_contact_pos

        if step % 100 == 0:
            box_pos = as_numpy(obj.get_pos())
            print(
                f"Shove:\n speed={commanded_speed:.3f}\n box_pos={box_pos}\n",
                f"fingertip_pos={contact_pos}\n target_v={target_velocity}\n",
                f"predicted_v={predicted_velocity}\n actual_delta={actual_delta}\n",
                f"qdot_scale={qdot_scale:.3f}\n qdot={qdot}",
            )

    for _ in range(settle_steps):
        robot.stop_velocity()
        step_once()