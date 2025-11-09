import logging

from rclpy.time import Time
import numpy as np
from scipy.spatial.transform import Rotation

import ros2_numpy as rnp
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile,
                       ReliabilityPolicy,
                       HistoryPolicy,
                       DurabilityPolicy)

from geometry_msgs.msg import (Vector3Stamped,
                               PoseStamped,
                               TwistStamped,
                               AccelStamped,
                               Point,
                               TransformStamped)
from traj_gen_interfaces.srv import AddPoint
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster, TransformBroadcaster
from traj_gen.min_snap_trajectory_generators import xyzMinDerivTrajectory
from traj_gen.waypoints import out as waypoints_out


class MinSnapTrajectoryGenerator(Node):
    """
    generate desired trajectory to be executed by the robot
    """

    def __init__(self):
        super().__init__('trajectory_generator')

        self.point_service = self.create_service(AddPoint, "add_point", self.add_point_callback)

        # setpoints publishers
        self.att_sp_rpy_pub = self.create_publisher(Vector3Stamped, '/target_att_rpy', 10)  # att in rpy degrees
        self.pose_sp_pub = self.create_publisher(PoseStamped, '/target_pose', 10)  # pose (xyz and quaternion)
        self.vel_sp_pub = self.create_publisher(TwistStamped, '/target_twist', 10)  # linear and angular velocity
        self.acc_sp_pub = self.create_publisher(AccelStamped, '/target_accel', 10)  # linear and angular acceleration
        self.jerk_sp_pub = self.create_publisher(Vector3Stamped, '/target_jerk', 10)  # linear jerk
        self.snap_sp_pub = self.create_publisher(Vector3Stamped, '/target_snap', 10)  # linear snap
        self.force_sp_pub = self.create_publisher(Float32, '/target_force', 10)
        self.pos = [0, 0, 0]
        self.pos_target = [0, 0, 0]
        # self.wrench_sp_pub = self.create_publisher(WrenchStamped,'/target_wrench', 10)

        # state subscribers (used by some traj generators such as yaw follower)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.att_rpy_sub = self.create_subscription(Vector3Stamped, '/att_rpy', self.rpy_callback, qos_profile)
        self.position_sub = self.create_subscription(Vector3Stamped, '/position', self.position_callback, qos_profile)

        # trajectory visualization in rviz
        self.waypoints_pub = self.create_publisher(Path, "/traj_gen/waypoints", 10)
        self.full_path_pub = self.create_publisher(Path, "/traj_gen/full_path", 10)
        self.timed_path_pub = self.create_publisher(Path, "/traj_gen/path", 10)
        self.vel_arrows_pub = self.create_publisher(Marker, "/traj_gen/vel_arrows", 10)

        self.frame_id_param = self.declare_parameter('frame_id', 'traj_gen')
        self.child_frame_id_param = self.declare_parameter('child_frame_id', 'traj_gen_node')
        self.declare_parameter('odom_frame', 'base_link')
        self.declare_parameter('target_frame', 'traj_gen_node')
        self.declare_parameter('reference_frame', 'world_ned')
        self.declare_parameter('hz', 120)

        self.transform_broadcaster = TransformBroadcaster(node=Node('traj_gen_tf_broadcaster'))

        self.xyz_waypoints = np.array(waypoints_out)
        self.rpy_waypoints = []
        self.waypoints_t = []
        # for rviz visualization
        self.position_path_msg = Path()  # plot trajectory
        self.position_path_msg.header.frame_id = self.frame_id
        self.trail_size = 1000  # maximum history to keep
        self.vel_heads = []
        self.vel_tails = []
        self.cur_odometry_position = np.array([0.0, 0.0, 0.0])
        self.cur_odometry_rotation = np.array([0.0, 0.0, 0.0, 0.0])
        self.next_point_treshold = 0.1
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.odom_clock = self.create_timer(1 / self.hz, self.cb_odom_frame)

        self.target_clock = self.create_timer(1 / self.hz, self.cb_target_frame)
        self.xyz_gen = xyzMinDerivTrajectory(self.xyz_waypoints)

        update_freq = 100.0  # hz
        self.start_time = self.get_clock().now().nanoseconds
        self.prev_time = 0.0
        self.update_callback_timer = self.create_timer(1.0 / update_freq, self.update_callback)

        # publish waypoints only once (if available)
        waypoints_msgs = Path()  # plot trajectory
        waypoints_msgs.header.frame_id = self.frame_id
        for position in self.xyz_waypoints:
            q_tmp = Rotation.from_euler(
                'XYZ', [0, 0, 0]).as_quat()
            q = np.zeros(4)
            q[0], q[1], q[2], q[3] = q_tmp[3], q_tmp[0], q_tmp[1], q_tmp[2]
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.child_frame_id
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            pose_msg.pose.orientation.w = q[0]
            pose_msg.pose.orientation.x = q[1]
            pose_msg.pose.orientation.y = q[2]
            pose_msg.pose.orientation.z = q[3]
            waypoints_msgs.poses.append(pose_msg)
            self.waypoints_pub.publish(waypoints_msgs)


    def add_point_callback(self, req, res):
        print("Add point callback")
        print(req, flush=True)
        self.xyz_gen.add_point(req.x, req.y, req.z)
        res.success = True
        return res

    def update_callback(self) -> None:
        t = ((self.get_clock().now().nanoseconds - self.start_time)
             / 1000_000_000.0)
        dt = (t - self.prev_time)
        self.prev_time = t

        # holders
        position, vel, _, att_rpy, = None, None, None, None

        # update translational motion
        position = self.xyz_gen.eval(position=np.array(self.pos),
                                     pos_target=np.array(self.pos_target),
                                     rotation=self.cur_odometry_rotation,
                                     treshhold=self.next_point_treshold)
        att_rpy = np.array([0, 0, 0])

        self.pub_pose_cmd(position, att_rpy)

        self.pub_transform_broadcaster(position, att_rpy)
        self.pub_rpy_cmd(att_rpy)

        waypoints_msgs = Path()  # plot trajectory
        waypoints_msgs.header.frame_id = self.frame_id
        for position in self.xyz_waypoints:
            q_tmp = Rotation.from_euler(
                'XYZ', [0, 0, 0]).as_quat()
            q = np.zeros(4)
            q[0], q[1], q[2], q[3] = q_tmp[3], q_tmp[0], q_tmp[1], q_tmp[2]
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.child_frame_id
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            pose_msg.pose.orientation.w = q[0]
            pose_msg.pose.orientation.x = q[1]
            pose_msg.pose.orientation.y = q[2]
            pose_msg.pose.orientation.z = q[3]
            waypoints_msgs.poses.append(pose_msg)
            self.waypoints_pub.publish(waypoints_msgs)

        # self.pub_linear_twist_cmd(vel)
        # self.pub_linear_accel_cmd(acc)
        # self.pub_linear_jerk_cmd(jerk)
        # self.pub_linear_snap_cmd(snap)

        # # ====== Publish trajectory visualization  ==========
        # if position is not None and att_rpy is not None:
        #     self.pub_traj_path(position, att_rpy)
        #     # Publish arrow markers for velocity if available
        #     if vel is not None:
        #         self.pub_vel_arrows(1, position, vel, dt)

    def rpy_callback(self, msg: Vector3Stamped) -> None:
        self.cur_rpy = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def position_callback(self, msg: Vector3Stamped) -> None:
        self.cur_position = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def odometry_callback(self, msg: Odometry) -> None:
        self.cur_odometry_position = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.cur_odometry_rotation = np.array(
            [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w])

    def cb_odom_frame(self) -> None:
        now = Time()
        try:
            t = self.tf_buffer.lookup_transform(self.odom_frame, self.reference_frame, now,
                                                rclpy.duration.Duration(seconds=0.5 / self.hz))

        except Exception as e:
            self.get_logger().error(f"{e}")
            return
        transform = rnp.numpify(t.transform)
        pos_from_tf = transform[:3, 3]
        self.pos = pos_from_tf

    def cb_target_frame(self) -> None:
        now = Time()
        try:
            t = self.tf_buffer.lookup_transform(self.target_frame, self.reference_frame, now,
                                                rclpy.duration.Duration(seconds=0.5 / self.hz))
        except Exception as e:
            self.get_logger().error(f"{e}")
            return
        transform = rnp.numpify(t.transform)
        pos_from_tf = transform[:3, 3]
        self.pos_target = pos_from_tf

    def pub_rpy_cmd(self, sig: np.ndarray) -> Vector3Stamped:
        # publish trajectory attitude in rpy representation in degrees
        # timestamp = int(self.get_clock().now().nanoseconds / 1000)
        rpy_msg = Vector3Stamped()
        rpy_msg.header.frame_id = self.child_frame_id
        rpy_msg.vector.x = np.rad2deg(sig[0])
        rpy_msg.vector.y = np.rad2deg(sig[1])
        rpy_msg.vector.z = np.rad2deg(sig[2])
        self.att_sp_rpy_pub.publish(rpy_msg)
        return rpy_msg

    def pub_linear_twist_cmd(self, linear_vel: np.ndarray) -> TwistStamped:
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = self.child_frame_id
        vel_msg.twist.linear.x = linear_vel[0]
        vel_msg.twist.linear.y = linear_vel[1]
        vel_msg.twist.linear.z = linear_vel[2]
        self.vel_sp_pub.publish(vel_msg)
        return vel_msg

    def pub_full_twist_cmd(self, linear_vel: np.ndarray, angular_vel: np.ndarray) -> TwistStamped:
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = self.child_frame_id
        vel_msg.twist.linear.x = linear_vel[0]
        vel_msg.twist.linear.y = linear_vel[1]
        vel_msg.twist.linear.z = linear_vel[2]
        vel_msg.twist.angular.x = angular_vel[0]
        vel_msg.twist.angular.y = angular_vel[1]
        vel_msg.twist.angular.z = angular_vel[2]

        self.vel_sp_pub.publish(vel_msg)
        return vel_msg

    def pub_linear_accel_cmd(self, linear_acc: np.ndarray) -> AccelStamped:
        acc_msg = AccelStamped()
        acc_msg.header.frame_id = self.child_frame_id
        acc_msg.accel.linear.x = linear_acc[0]
        acc_msg.accel.linear.y = linear_acc[1]
        acc_msg.accel.linear.z = linear_acc[2]
        self.acc_sp_pub.publish(acc_msg)
        return acc_msg

    def pub_full_accel_cmd(self, linear_acc: np.ndarray, angular_acc: np.ndarray) -> AccelStamped:
        acc_msg = AccelStamped()
        acc_msg.header.frame_id = self.child_frame_id
        acc_msg.accel.linear.x = linear_acc[0]
        acc_msg.accel.linear.y = linear_acc[1]
        acc_msg.accel.linear.z = linear_acc[2]
        acc_msg.accel.angular.x = angular_acc[0]
        acc_msg.accel.angular.y = angular_acc[1]
        acc_msg.accel.angular.z = angular_acc[2]

        self.acc_sp_pub.publish(acc_msg)
        return acc_msg

    def pub_linear_jerk_cmd(self, linear_jerk: np.ndarray) -> Vector3Stamped:
        msg = Vector3Stamped()
        msg.header.frame_id = self.child_frame_id
        msg.vector.x = linear_jerk[0]
        msg.vector.y = linear_jerk[1]
        msg.vector.z = linear_jerk[2]
        self.jerk_sp_pub.publish(msg)
        return msg

    def pub_linear_snap_cmd(self, linear_snap: np.ndarray) -> Vector3Stamped:
        msg = Vector3Stamped()
        msg.header.frame_id = self.child_frame_id
        msg.vector.x = linear_snap[0]
        msg.vector.y = linear_snap[1]
        msg.vector.z = linear_snap[2]
        self.snap_sp_pub.publish(msg)
        return msg

    def pub_pose_cmd(self, position: np.ndarray, rpy: np.ndarray) -> PoseStamped:
        q_tmp = Rotation.from_euler(
            'XYZ', [rpy[0], rpy[1], rpy[2]]).as_quat()
        q = np.zeros(4)
        q[0] = q_tmp[3]
        q[1] = q_tmp[0]
        q[2] = q_tmp[1]
        q[3] = q_tmp[2]
        msg = PoseStamped()
        msg.header.frame_id = self.child_frame_id
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.w = q[0]
        msg.pose.orientation.x = q[1]
        msg.pose.orientation.y = q[2]
        msg.pose.orientation.z = q[3]
        self.pose_sp_pub.publish(msg)
        return msg

    def pub_force_cmd(self, force: float) -> None:
        msg = Float32()
        msg.data = force
        self.force_sp_pub.publish(msg)
        return msg

    def pub_traj_path(self, position, att_rpy):
        q_tmp = Rotation.from_euler(
            'XYZ', [att_rpy[0], att_rpy[1], att_rpy[2]]).as_quat()
        q = np.zeros(4)
        q[0], q[1], q[2], q[3] = q_tmp[3], q_tmp[0], q_tmp[1], q_tmp[2]
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.w = q[0]
        pose_msg.pose.orientation.x = q[1]
        pose_msg.pose.orientation.y = q[2]
        pose_msg.pose.orientation.z = q[3]
        self.position_path_msg.poses.append(pose_msg)
        if len(self.position_path_msg.poses) > self.trail_size:
            del self.position_path_msg.poses[0]
        self.timed_path_pub.publish(self.position_path_msg)

    def pub_vel_arrows(self, id, position, vel, dt):
        # append position to the tails array and limit array size
        tail_point = Point()
        tail_point.x = position[0]
        tail_point.y = position[1]
        tail_point.z = position[2]
        self.vel_tails.append(tail_point)
        if len(self.vel_tails) > self.trail_size:
            del self.vel_tails[0]

        # append position to the heads array and limit array size
        head = position + vel * dt
        head_point = Point()
        head_point.x = head[0]
        head_point.y = head[1]
        head_point.z = head[2]
        self.vel_heads.append(head_point)
        if len(self.vel_heads) > self.trail_size:
            del self.vel_heads[0]

        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = self.child_frame_id
        msg.ns = "arrows"
        msg.id = id
        msg.type = Marker.LINE_LIST
        msg.scale.x = 0.2
        msg.scale.y = 0.2
        msg.scale.z = 0.0
        msg.color.r = 0.8
        msg.color.g = 0.2
        msg.color.b = 0.4
        msg.color.a = 0.9
        for tail, head in zip(self.vel_tails, self.vel_heads):
            msg.points.append(tail)
            msg.points.append(head)
        self.vel_arrows_pub.publish(msg)

    def pub_transform_broadcaster(self, position: np.ndarray, rpy: np.ndarray) -> None:
        q_tmp = Rotation.from_euler(
            'XYZ', [rpy[0], rpy[1], rpy[2]]
        ).as_quat()
        q = np.zeros(4)
        q[0] = q_tmp[3]
        q[1] = q_tmp[0]
        q[2] = q_tmp[1]
        q[3] = q_tmp[2]
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.transform.translation.x = position[0]
        msg.transform.translation.y = position[1]
        msg.transform.translation.z = position[2]
        msg.transform.rotation.w = q[0]
        msg.transform.rotation.x = q[1]
        msg.transform.rotation.y = q[2]
        msg.transform.rotation.z = q[3]

        self.transform_broadcaster.sendTransform(msg)

    @property
    def frame_id(self) -> str:
        return self.get_parameter('frame_id').value

    @property
    def child_frame_id(self) -> str:
        return self.get_parameter('child_frame_id').value

    @property
    def reference_frame(self) -> str:
        return self.get_parameter('reference_frame').value

    @property
    def hz(self) -> int:
        return self.get_parameter('hz').value

    @property
    def odom_frame(self) -> str:
        return self.get_parameter('odom_frame').value

    @property
    def target_frame(self) -> str:
        return self.get_parameter('target_frame').value


def main(args=None):
    """Main function to execute"""
    rclpy.init(args=args)

    sig_generator = MinSnapTrajectoryGenerator()

    rclpy.spin(sig_generator)
    sig_generator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
