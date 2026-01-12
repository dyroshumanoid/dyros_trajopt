# utils/sideflip_util.py
import numpy as np
import pinocchio
import crocoddyl
from typing import List, Dict, Optional
from scipy.spatial.transform import Rotation as R


class SideflipProblem:
    def __init__(
        self,
        robot_model: pinocchio.Model,
        base_frame_name: str,
        rf_contact_frame_name: str,
        lf_contact_frame_name: str,
        integrator: str = "rk4",
        control: str = "zero",
        weights: Dict = None,
        flyup_roll_target: float = -np.pi, 
        land_roll_target: float = -2*np.pi,
        foot_sep: float = 0.09,
    ):
        """
        Construct a sideflip problem.

        Required referenceConfigurations (SRDF):
            "side_sitting",
            "side_takeoff",
            "side_flying",
            "side_landing"
        """
        self.w = weights if weights else {}
        self.robot_model = robot_model
        self.robot_data = robot_model.createData()
        self.state = crocoddyl.StateMultibody(self.robot_model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self._integrator = integrator
        self._control = control

        self.flyup_roll_target = float(flyup_roll_target)
        self.land_roll_target = float(land_roll_target)
        self.foot_sep = float(foot_sep)

        # Get frame IDs of base and foot contact frames
        self.base_frame_id = self.robot_model.getFrameId(base_frame_name)
        self.lf_contact_frame_id = self.robot_model.getFrameId(lf_contact_frame_name)
        self.rf_contact_frame_id = self.robot_model.getFrameId(rf_contact_frame_name)

        # Default states
        self.x_ground = np.concatenate(
            [
                self.robot_model.referenceConfigurations["side_sitting"].copy(),
                np.zeros(self.robot_model.nv)
            ]
        )
        self.x_takeoff = np.concatenate(
            [
                self.robot_model.referenceConfigurations["side_takeoff"].copy(),
                np.zeros(self.robot_model.nv)
            ]
        )
        self.x_flying = np.concatenate(
            [
                self.robot_model.referenceConfigurations["side_flying"].copy(),
                np.zeros(self.robot_model.nv)
            ]
        )
        self.x_inverse = np.concatenate(
            [
                self.robot_model.referenceConfigurations["side_inverse"].copy(),
                np.zeros(self.robot_model.nv)
            ]
        )
        self.x_landing = np.concatenate(
            [
                self.robot_model.referenceConfigurations["side_landing"].copy(),
                np.zeros(self.robot_model.nv)
            ]
        )

        # Define friction coefficient and ground
        self.mu = 0.7
        self.R_ground = np.eye(3)

        # Weights for joint state bounds cost
        self.x_bounds_weights = np.array(
            [0.0] * 6 # base SE3 residual (no bounds)
            + [100.0] * (self.robot_model.nv - 6) # joint position residual
            + [0.0] * 3 # base linear velocity residual (no bounds)
            + [100.0] * 3 # base angular velocity residual
            + [100.0] * (self.robot_model.nv - 6) # joint velocity residual
        )
    
    # =========================================================================
    # Stage 1: [pre_takeoff -> takeoff -> flyup -> inverse]
    # =========================================================================
    def create_sideflip_problem_first_stage(
        self,
        x0: np.ndarray,
        jump_height: float,
        jump_length: List[float],
        dt: float,
        num_ground_knots: int,
        num_flying_knots: int,
        v_liftoff: float,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
        v_roll_kick: float,
    ) -> crocoddyl.ShootingProblem:
        """
        Construct a shooting problem for backflip first stage.

        The first stage includes the following phases:
            [Ready, Take-Off, Flying-Up]

        The robot body pitch rotates from 0 to -135 degrees.

        :param x0: Initial state of the robot (q0, v0).
        :param jump_height: Height of the jump in meters.
        :param jump_length: Length of the jump in meters (x, y, z).
        :param dt: Time step length in second.
        :param num_ground_knots: Number of knots on the ground.
        :param num_flying_knots: Number of knots in the air.
        :param v_liftoff: Lift-off velocity in the z direction.
        :param x_lb: Lower bounds for joint states.
        :param x_ub: Upper bounds for joint states.

        :return: A shooting problem for backflip first stage.
        """
        # ---------------------------------------------------------------------------- #
        # Initialize ----------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        sideflip_action_models = []
        q0 = x0[: self.robot_model.nq]
        pinocchio.forwardKinematics(self.robot_model, self.robot_data, q0)
        pinocchio.updateFramePlacements(self.robot_model, self.robot_data)

        # Initial foot contact frames reference SE3
        lf_contact_pose_0 = self.robot_data.oMf[self.lf_contact_frame_id]
        rf_contact_pose_0 = self.robot_data.oMf[self.rf_contact_frame_id]
        foot_poses_ref_0 = {
            self.lf_contact_frame_id: lf_contact_pose_0,
            self.rf_contact_frame_id: rf_contact_pose_0,
        }
        base_pos_ref_0 = (
            lf_contact_pose_0.translation + rf_contact_pose_0.translation
        ) / 2.0
        base_pos_ref_0[2] = self.robot_data.oMf[self.base_frame_id].translation[2]

        # --------------------------------------------------------------------- #
        # 1. Take-Off Phase: x0 -> side_takeoff
        # --------------------------------------------------------------------- #

        x_track_weights_takeoff = np.array(
            self.w.get("side_takeoff_base_pos", [25.0, 100.0, 0.0])
            + self.w.get("side_takeoff_base_rot", [100.0, 0.0, 100.0])
            + self.w.get("side_takeoff_joint", [50.0]) * (self.state.nv - 6)
            + self.w.get("side_takeoff_base_v", [25.0, 100.0, 0.0])
            + self.w.get("side_takeoff_base_w", [100.0, 0.0, 100.0])
            + self.w.get("side_takeoff_joint_v", [0.0]) * (self.state.nv - 6)
        )

        x_takeoff = self.x_takeoff.copy() #수정함
        x_takeoff[self.robot_model.nq : self.robot_model.nq + 6] = np.array(
            [0.0, 0.0, v_liftoff, v_roll_kick, 0.0, 0.0]
        )
        x_ref_traj_takeoff = self.state_interp(x0, x_takeoff, num_ground_knots)

        take_off = [
            self.create_knot_action_model(
                dt, 
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                foot_poses_ref=foot_poses_ref_0,
                x_ref=x_ref_traj_takeoff[k],
                x_track_weights=x_track_weights_takeoff,
                x_track_cost_weight=1e3,
                x_lb=x_lb, 
                x_ub=x_ub, 
                x_bounds_cost_weight=1e12,
            )
            for k in range(num_ground_knots)
        ]
        take_off[-1] = self.create_knot_action_model(
            dt, [self.lf_contact_frame_id, self.rf_contact_frame_id],
            foot_poses_ref=foot_poses_ref_0,
            x_ref=x_takeoff,
            x_track_weights=np.array(
                self.w.get("side_takeoff_terminal_base_pos", [25.0, 100.0, 100.0])
                + self.w.get("side_takeoff_terminal_base_rot", [100.0, 0.0, 100.0])
                + self.w.get("side_takeoff_terminal_joint", [50.0]) * (self.state.nv - 6)
                + self.w.get("side_takeoff_terminal_base_v", [0.0, 0.0, 200.0])
                + self.w.get("side_takeoff_terminal_base_w", [100.0, 0.0, 100.0])
                + self.w.get("side_takeoff_terminal_joint_v", [0.0]) * (self.state.nv - 6)
            ),
            x_track_cost_weight=1e6,
            x_lb=x_lb, 
            x_ub=x_ub, 
            x_bounds_cost_weight=1e12,
        )
# ---------------------------------------------------------------------------- #
        # 2. Flying-Up Phase: [Takeoff -> Flying(Tuck)]
        # ---------------------------------------------------------------------------- #
        base_pos_ref_0 = x_takeoff[:3]
        base_pos_ref_peak = base_pos_ref_0 + np.array(
            [
                jump_length[0] / 2.0,
                jump_length[1] / 2.0,
                jump_height, 
            ]
        )
        base_pos_ref_traj = [
            base_pos_ref_0 + (base_pos_ref_peak - base_pos_ref_0) * (k + 1) / num_flying_knots
            for k in range(num_flying_knots)
        ]

        start_roll = R.from_quat(x_takeoff[3:7]).as_euler("zyx")[2]
        target_roll = self.flyup_roll_target
        
        base_roll_ref_traj = [
            start_roll + (target_roll - start_roll) * (k + 1) / num_flying_knots
            for k in range(num_flying_knots)
        ]
        
        base_quat_ref_traj = [
            R.from_euler("zyx", [0.0, 0.0, base_roll_ref_traj[k]]).as_quat()
            for k in range(num_flying_knots)
        ]

        base_roll_ang_vel_ref = (target_roll - start_roll) / (num_flying_knots * dt)
        base_ang_vel_ref = [
            np.array([base_roll_ang_vel_ref, 0.0, 0.0])
            for k in range(num_flying_knots)
        ]

        n_interp = max(0, num_flying_knots - 10)
        x_ref_traj_flyup = np.vstack(
            [
                self.state_interp(x_takeoff, self.x_flying, n_interp),
                np.tile(self.x_flying, (num_flying_knots - n_interp, 1)),
            ]
        )

        x_ref_traj_flyup[:, :3] = np.array(base_pos_ref_traj) 
        x_ref_traj_flyup[:, 3:7] = np.array(base_quat_ref_traj)
        
        x_ref_traj_flyup[:, self.robot_model.nq + 3 : self.robot_model.nq + 6] = base_ang_vel_ref


        x_track_weights_flyup = np.array(
            self.w.get("side_flyup_base_pos", [10.0, 100.0, 0.0])
            + self.w.get("side_flyup_base_rot", [100.0, 10.0, 100.0])
            + self.w.get("side_flyup_joint", [75.0]) * (self.state.nv - 6)
            + self.w.get("side_flyup_base_v", [0.0, 100.0, 0.0])
            + self.w.get("side_flyup_base_w", [100.0, 100.0, 100.0])
            + self.w.get("side_flyup_joint_v", [0.5]) * (self.state.nv - 6)
        )

        fly_up = [
            self.create_knot_action_model(
                dt, [], x_ref=x_ref_traj_flyup[k],
                x_track_weights=x_track_weights_flyup,
                x_track_cost_weight=1e3,
                x_lb=x_lb, x_ub=x_ub, x_bounds_cost_weight=1e12,
            )
            for k in range(num_flying_knots)
        ]
        
        fly_up[-1] = self.create_knot_action_model(
            dt, [], x_ref=x_ref_traj_flyup[-1],
            x_track_weights=np.array(
                self.w.get("side_flyup_terminal_base_pos", [10.0, 100.0, 0.0])
                + self.w.get("side_flyup_terminal_base_rot", [1000.0, 100.0, 100.0])
                + self.w.get("side_flyup_terminal_joint", [150.0]) * (self.state.nv - 6)
                + self.w.get("side_flyup_terminal_base_v", [0.0, 100.0, 50.0])
                + self.w.get("side_flyup_terminal_base_w", [1000.0, 100.0, 100.0])
                + self.w.get("side_flyup_terminal_joint_v", [0.0]) * (self.state.nv - 6)
            ),
            x_track_cost_weight=1e6,
            x_lb=x_lb, x_ub=x_ub, x_bounds_cost_weight=1e12,
        )

        sideflip_action_models += take_off
        sideflip_action_models += fly_up
        return crocoddyl.ShootingProblem(x0, sideflip_action_models[:-1], sideflip_action_models[-1])

    # =========================================================================
    # Stage 2: [flydown -> impulse landing -> landed]
    # =========================================================================
    def create_sideflip_problem_second_stage(
        self,
        x0: np.ndarray,
        jump_length: List[float],
        dt: float,
        num_ground_knots: int,
        num_flying_knots: int,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
    ) -> crocoddyl.ShootingProblem:
        """
        Second stage:
            [Flying-Down (side_untuck -> side_landing), Landing (impulse), Landed (side_sitting)]
        Roll returns to land_roll_target.
        """
        sideflip_action_models = []

        pinocchio.forwardKinematics(
            self.robot_model, self.robot_data, self.x_ground[: self.robot_model.nq]
        )
        pinocchio.updateFramePlacements(self.robot_model, self.robot_data)

        landing_x = float(jump_length[0])
        landing_y = float(jump_length[1])
        foot_poses_ref_final = {
            self.lf_contact_frame_id: pinocchio.SE3(
                np.eye(3), np.array([landing_x, landing_y+ self.foot_sep, 0.0])
            ),
            self.rf_contact_frame_id: pinocchio.SE3(
                np.eye(3), np.array([landing_x, landing_y- self.foot_sep, 0.0])
            ),
        }


        # ---------------------------------------------------------------------------- #
        # 1. Flying-Down Phase: [Inverse -> Landing]
        # ---------------------------------------------------------------------------- #
        self.x_landing[:3] += np.array(jump_length)
        base_roll_0 = R.from_quat(x0[3:7]).as_euler("zyx")[2]
        base_roll_f = R.from_quat(self.x_landing[3:7]).as_euler("zyx")[2]

        if base_roll_0 > 0: base_roll_0 -= 2.0 * np.pi
        while base_roll_f > base_roll_0: base_roll_f -= 2.0 * np.pi

        base_roll_ref_traj = [
            base_roll_0 + (base_roll_f - base_roll_0) * (k + 1) / num_flying_knots
            for k in range(num_flying_knots)
        ]
        base_quat_ref_traj = [
            R.from_euler("zyx", [0.0, 0.0, base_roll_ref_traj[k]]).as_quat()
            for k in range(num_flying_knots)
        ]
        base_roll_ang_vel_ref = (base_roll_f - base_roll_0) / (num_flying_knots * dt)

        x_ref_traj_flydown = self.state_interp(x0, self.x_landing, num_flying_knots)
        x_ref_traj_flydown[:, 3:7] = np.array(base_quat_ref_traj)
        
        x_ref_traj_flydown[:, self.robot_model.nq : self.robot_model.nq + 3] = 0.0
        x_ref_traj_flydown[:, self.robot_model.nq + 3] = base_roll_ang_vel_ref * np.ones(num_flying_knots)
        x_ref_traj_flydown[:, self.robot_model.nq + 4] = np.zeros(num_flying_knots)
        x_ref_traj_flydown[:, self.robot_model.nq + 5] = np.zeros(num_flying_knots)

        joint_pos_0 = x0[7 : self.robot_model.nq]
        joint_pos_ref_land = self.x_landing[7 : self.robot_model.nq]
        joint_pos_ref_traj = [
            joint_pos_0
            + (joint_pos_ref_land - joint_pos_0) * (k + 1) / (num_flying_knots - 12)
            for k in range(num_flying_knots - 12)
        ] + [joint_pos_ref_land] * 12
        x_ref_traj_flydown[:, 7 : self.robot_model.nq] = np.array(joint_pos_ref_traj)
        x_track_weights_flydown = np.array(
            self.w.get("side_flydown_base_pos", [30.0, 0.0, 0.0])
            + self.w.get("side_flydown_base_rot", [10.0, 10.0, 10.0])
            + self.w.get("side_flydown_joint", [120.0]) * (self.state.nv - 6)
            + self.w.get("side_flydown_base_v", [0.0, 0.0, 0.0])
            + self.w.get("side_flydown_base_w", [0.0, 0.0, 0.0])
            + self.w.get("side_flydown_joint_v", [0.1]) * (self.state.nv - 6)
        )

        fly_down = [
            self.create_knot_action_model(
                dt, [], foot_poses_ref=None,
                x_ref=x_ref_traj_flydown[k],
                x_track_weights=x_track_weights_flydown,
                x_track_cost_weight=1e4,
                x_lb=x_lb, x_ub=x_ub, x_bounds_cost_weight=1e12,
            )
            for k in range(num_flying_knots)
        ]

        fly_down[-1] = self.create_knot_action_model(
            dt, [], foot_poses_ref=foot_poses_ref_final,
            track_foot_poses=True,
            foot_poses_track_cost_weight=1e8,
            x_ref=x_ref_traj_flydown[-1],
            x_track_weights=np.array(
                self.w.get("side_flydown_terminal_base_pos", [80.0, 0.0, 100.0])
                + self.w.get("side_flydown_terminal_base_rot", [10.0, 10.0, 10.0])
                + self.w.get("side_flydown_terminal_joint", [120.0]) * (self.state.nv - 6)
                + self.w.get("side_flydown_terminal_base_v", [0.0, 0.0, 0.0])
                + self.w.get("side_flydown_terminal_base_w", [0.0, 0.0, 0.0])
                + self.w.get("side_flydown_terminal_joint_v", [0.0]) * (self.state.nv - 6)
            ),
            x_track_cost_weight=1e6,
            x_lb=x_lb, x_ub=x_ub, x_bounds_cost_weight=1e12,
        )
        # ----------------------------------2----------------------------------- #
        # Landing (impulse)
        # --------------------------------------------------------------------- #
        
        landing = [
            self.create_knot_impulse_model(
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                foot_poses_ref=foot_poses_ref_final,
                foot_poses_track_cost_weight=1e6,
                x_ref=x_ref_traj_flydown[-1],
                x_track_weights=np.array(
                    self.w.get("side_landing_base_pos", [0.0, 0.0, 0.0])
                    + self.w.get("side_landing_base_rot", [10.0, 10.0, 10.0])
                    + self.w.get("side_landing_joint", [120.0]) * (self.state.nv - 6)
                    + self.w.get("side_landing_base_v", [0.0, 0.0, 0.0])
                    + self.w.get("side_landing_base_w", [0.0, 0.0, 0.0])
                    + self.w.get("side_landing_joint_v", [0.0]) * (self.state.nv - 6)
                ),
                x_track_cost_weight=1e3,
            )
        ]

        # --------------------------------------------------------------------- #
        # Landed stabilization: track side_sitting with shifted y
        # --------------------------------------------------------------------- #
        x_at_impact = x_ref_traj_flydown[-1].copy()
        x_ref_final = self.x_ground.copy()
        x_ref_final[0:3] = [landing_x, landing_y, self.x_ground[2]]
        x_landed_traj = self.state_interp(x_at_impact, x_ref_final, num_ground_knots)
        
        landed = [
            self.create_knot_action_model(
                dt,
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                foot_poses_ref=foot_poses_ref_final,
                x_ref=x_landed_traj[k],
                x_track_weights=np.array(
                    self.w.get("side_landed_base_pos", [120.0, 120.0, 120.0])
                    + self.w.get("side_landed_base_rot", [50.0, 50.0, 50.0])
                    + self.w.get("side_landed_joint", [100.0]) * (self.state.nv - 6)
                    + self.w.get("side_landed_base_v", [0.0, 0.0, 0.0])
                    + self.w.get("side_landed_base_w", [0.0, 0.0, 0.0])
                    + self.w.get("side_landed_joint_v", [5.0]) * (self.state.nv - 6)
                ),
                x_track_cost_weight=1e3,
            )
            for k in range(num_ground_knots)
        ]

        sideflip_action_models += fly_down
        sideflip_action_models += landing
        sideflip_action_models += landed
        return crocoddyl.ShootingProblem(x0, sideflip_action_models[:-1], sideflip_action_models[-1])

    # =========================================================================
    # Knot Models (same as your original, no omissions)
    # =========================================================================
    def create_knot_action_model(
        self,
        dt: float,
        support_foot_ids: List[int],
        wrench_cone_cost_weight: float = 1e1,
        foot_poses_ref: Dict[int, pinocchio.SE3] = None,
        track_foot_poses: bool = False,
        foot_poses_track_cost_weight: float = 1e6,
        x_ref: np.ndarray = None,
        x_track_weights: np.ndarray = None,
        x_track_cost_weight: float = None,
        x_lb: np.ndarray = None,
        x_ub: np.ndarray = None,
        x_bounds_cost_weight: float = 1e6,
        ctrl_regu_cost_weight: float = 1e-8,
    ) -> crocoddyl.IntegratedActionModelAbstract:

        if len(support_foot_ids) > 0 and foot_poses_ref is None:
            raise ValueError("foot_poses_ref must be provided for support contacts.")
        if track_foot_poses:
            if foot_poses_ref is None:
                raise ValueError("foot_poses_ref must be provided when track_foot_poses=True.")
            if foot_poses_track_cost_weight is None:
                raise ValueError("foot_poses_track_cost_weight must be provided.")
        if x_ref is not None:
            if x_track_weights is None:
                raise ValueError("x_track_weights must be provided.")
            if x_track_cost_weight is None:
                raise ValueError("x_track_cost_weight must be provided.")

        nu = self.actuation.nu
        costs = crocoddyl.CostModelSum(self.state, nu)

        # Contacts
        contact_models = crocoddyl.ContactModelMultiple(self.state, nu)
        for foot_id in support_foot_ids:
            foot_contact_model = crocoddyl.ContactModel6D(
                self.state,
                foot_id,
                foot_poses_ref[foot_id],
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 30.0]),
            )
            contact_models.addContact(self.robot_model.frames[foot_id].name + "_contact", foot_contact_model)

        # Wrench cone costs
        foot_size = np.array([0.1, 0.05])  # [length, width]
        for foot_id in support_foot_ids:
            cone = crocoddyl.WrenchCone(self.R_ground, self.mu, foot_size)
            wrench_residual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, foot_id, cone, nu, fwddyn=True
            )
            wrench_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrench_cone_cost = crocoddyl.CostModelResidual(self.state, wrench_activation, wrench_residual)
            costs.addCost(
                self.robot_model.frames[foot_id].name + "_wrench_cone",
                wrench_cone_cost,
                wrench_cone_cost_weight,
            )

        # Foot pose tracking costs
        if track_foot_poses:
            for foot_id, foot_pose in foot_poses_ref.items():
                foot_poses_track_residual = crocoddyl.ResidualModelFramePlacement(
                    self.state, foot_id, foot_pose, nu
                )
                foot_poses_track_cost = crocoddyl.CostModelResidual(self.state, foot_poses_track_residual)
                costs.addCost(
                    self.robot_model.frames[foot_id].name + "_foot_poses_track",
                    foot_poses_track_cost,
                    foot_poses_track_cost_weight,
                )

        # State tracking
        if x_ref is not None:
            x_track_residual = crocoddyl.ResidualModelState(self.state, x_ref, nu)
            x_track_activation = crocoddyl.ActivationModelWeightedQuad(x_track_weights ** 2)
            x_track_cost = crocoddyl.CostModelResidual(self.state, x_track_activation, x_track_residual)
            costs.addCost("x_track", x_track_cost, x_track_cost_weight)

        # State bounds (barrier)
        if x_lb is not None and x_ub is not None:
            x_bounds_residual = crocoddyl.ResidualModelState(self.state, x_lb, nu)

            r_lb = np.zeros(2 * self.robot_model.nv)
            r_ub = self.state.diff(x_lb, x_ub)
            r_ub[3:6] = 1e6 * np.ones(3)  # base orientation residuals huge

            x_bounds_activation = crocoddyl.ActivationModelWeightedQuadraticBarrier(
                crocoddyl.ActivationBounds(r_lb, r_ub),
                self.x_bounds_weights,
            )
            x_bounds_cost = crocoddyl.CostModelResidual(self.state, x_bounds_activation, x_bounds_residual)
            costs.addCost("x_bounds", x_bounds_cost, x_bounds_cost_weight)

        # Control regularization
        ctrl_residual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrl_regu_cost = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        costs.addCost("control_regularization", ctrl_regu_cost, ctrl_regu_cost_weight)

        # Dynamics
        dyn_model = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contact_models, costs, 0.0, True
        )

        # Control parametrization
        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        # Integrator
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dyn_model, control, dt)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(dyn_model, control, crocoddyl.RKType.four, dt)
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(dyn_model, control, crocoddyl.RKType.three, dt)
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(dyn_model, control, crocoddyl.RKType.two, dt)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dyn_model, control, dt)

        return model

    def create_knot_impulse_model(
        self,
        support_foot_ids: List[int],
        foot_poses_ref: Dict[int, pinocchio.SE3],
        x_ref: np.ndarray = None,
        foot_poses_track_cost_weight: float = 1e6,
        x_track_weights: np.ndarray = None,
        x_track_cost_weight: float = 1e3,
        r_coeff: float = 0.0,
        JMinvJt_damping: float = 1e-12,
    ) -> crocoddyl.ActionModelImpulseFwdDynamics:

        if x_ref is not None and x_track_weights is None:
            raise ValueError("x_track_weights must be provided when x_ref is provided.")

        impulse_models = crocoddyl.ImpulseModelMultiple(self.state)
        for foot_id in support_foot_ids:
            foot_impulse_model = crocoddyl.ImpulseModel6D(
                self.state, foot_id, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulse_models.addImpulse(self.robot_model.frames[foot_id].name + "_impulse", foot_impulse_model)

        nu = 0
        costs = crocoddyl.CostModelSum(self.state, nu)

        for foot_id, foot_pose in foot_poses_ref.items():
            foot_poses_track_residual = crocoddyl.ResidualModelFramePlacement(
                self.state, foot_id, foot_pose, nu
            )
            foot_poses_track_cost = crocoddyl.CostModelResidual(self.state, foot_poses_track_residual)
            costs.addCost(
                self.robot_model.frames[foot_id].name + "_foot_poses_track",
                foot_poses_track_cost,
                foot_poses_track_cost_weight,
            )

        x_track_residual = crocoddyl.ResidualModelState(self.state, x_ref, 0)
        x_track_activation = crocoddyl.ActivationModelWeightedQuad(x_track_weights ** 2)
        x_track_cost = crocoddyl.CostModelResidual(self.state, x_track_activation, x_track_residual)
        costs.addCost("x_track", x_track_cost, x_track_cost_weight)

        impulse_model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulse_models, costs, r_coeff, JMinvJt_damping
        )
        return impulse_model

    # def state_interp(self, x0: np.ndarray, xf: np.ndarray, N: int, type: str = "linear") -> np.ndarray:
    #     """
    #     Linear interpolation between x0 and xf (Euler in zyx).
    #     Returns array shape (N, nq+nv)
    #     """
    #     q0 = x0[: self.robot_model.nq]
    #     v0 = x0[self.robot_model.nq :]
    #     base_pos_0 = q0[:3]
    #     base_quat_0 = q0[3:7]
    #     base_eul_0 = R.from_quat(base_quat_0).as_euler("zyx")
    #     joint_pos_0 = q0[7:]
    #     base_vel_0 = v0[:6]
    #     joint_vel_0 = v0[6:]

    #     qf = xf[: self.robot_model.nq]
    #     vf = xf[self.robot_model.nq :]
    #     base_pos_f = qf[:3]
    #     base_quat_f = qf[3:7]
    #     base_eul_f = R.from_quat(base_quat_f).as_euler("zyx")
    #     joint_pos_f = qf[7:]
    #     base_vel_f = vf[:6]
    #     joint_vel_f = vf[6:]

    #     if N < 2:
    #         return np.tile(xf, (max(N, 1), 1))

    #     if type != "linear":
    #         raise ValueError(f"Interpolation type {type} is not supported.")

    #     base_pos_traj = [base_pos_0 + (base_pos_f - base_pos_0) * k / (N - 1) for k in range(N)]
    #     base_eul_traj = [base_eul_0 + (base_eul_f - base_eul_0) * k / (N - 1) for k in range(N)]
    #     base_quat_traj = [R.from_euler("zyx", base_eul_traj[k]).as_quat() for k in range(N)]
    #     joint_pos_traj = [joint_pos_0 + (joint_pos_f - joint_pos_0) * k / (N - 1) for k in range(N)]
    #     base_vel_traj = [base_vel_0 + (base_vel_f - base_vel_0) * k / (N - 1) for k in range(N)]
    #     joint_vel_traj = [joint_vel_0 + (joint_vel_f - joint_vel_0) * k / (N - 1) for k in range(N)]

    #     x_traj = np.hstack(
    #         [
    #             np.array(base_pos_traj),
    #             np.array(base_quat_traj),
    #             np.array(joint_pos_traj),
    #             np.array(base_vel_traj),
    #             np.array(joint_vel_traj),
    #         ]
    #     )
    #     return x_traj
    def state_interp(self, x0: np.ndarray, xf: np.ndarray, N: int, type: str = "linear") -> np.ndarray:
            """
            Linear interpolation between x0 and xf (Euler in zyx).
            Returns array shape (N, nq+nv)
            """
            q0 = x0[: self.robot_model.nq]
            v0 = x0[self.robot_model.nq :]
            base_pos_0 = q0[:3]
            base_quat_0 = q0[3:7]
            base_eul_0 = R.from_quat(base_quat_0).as_euler("zyx").copy() # .copy() 추가 (값 보호)
            joint_pos_0 = q0[7:]
            base_vel_0 = v0[:6]
            joint_vel_0 = v0[6:]

            qf = xf[: self.robot_model.nq]
            vf = xf[self.robot_model.nq :]
            base_pos_f = qf[:3]
            base_quat_f = qf[3:7]
            base_eul_f = R.from_quat(base_quat_f).as_euler("zyx").copy()
            joint_pos_f = qf[7:]
            base_vel_f = vf[:6]
            joint_vel_f = vf[6:]


            if base_eul_0[2] > 0:
                base_eul_0[2] -= 2.0 * np.pi
                

            while base_eul_f[2] > base_eul_0[2]:
                base_eul_f[2] -= 2.0 * np.pi

            if N < 2:
                return np.tile(xf, (max(N, 1), 1))

            if type != "linear":
                raise ValueError(f"Interpolation type {type} is not supported.")

            base_pos_traj = [base_pos_0 + (base_pos_f - base_pos_0) * k / (N - 1) for k in range(N)]
            base_eul_traj = [base_eul_0 + (base_eul_f - base_eul_0) * k / (N - 1) for k in range(N)]
            base_quat_traj = [R.from_euler("zyx", base_eul_traj[k]).as_quat() for k in range(N)]
            joint_pos_traj = [joint_pos_0 + (joint_pos_f - joint_pos_0) * k / (N - 1) for k in range(N)]
            base_vel_traj = [base_vel_0 + (base_vel_f - base_vel_0) * k / (N - 1) for k in range(N)]
            joint_vel_traj = [joint_vel_0 + (joint_vel_f - joint_vel_0) * k / (N - 1) for k in range(N)]

            x_traj = np.hstack(
                [
                    np.array(base_pos_traj),
                    np.array(base_quat_traj),
                    np.array(joint_pos_traj),
                    np.array(base_vel_traj),
                    np.array(joint_vel_traj),
                ]
            )
            return x_traj