import time
import math
import numpy as np
from mbot_bridge.api import MBot

# ==================== CONFIGURATION ====================

# Goal tolerance - how close we need to get to the goal
GOAL_XY_TOL = 0.10      # meters - within 10cm of goal position
GOAL_THETA_TOL = 0.15   # radians - within ~8.6 degrees of goal angle

# Hit the spot control gains
KP_LINEAR = 0.8         # proportional gain for linear speed
KP_ANGULAR = 1.5        # proportional gain for angular speed

# Speed limits for driving to goal
V_MAX = 0.40            # max linear speed (m/s)
W_MAX = 1.2             # max angular speed (rad/s)
V_MIN = 0.08            # minimum speed to overcome friction

# Obstacle detection
OBS_DETECT_DIST = 0.50  # if something closer than this in our path -> obstacle detected
OBS_CLEAR_DIST = 0.70   # when path clears beyond this -> continue to goal
SLICE_ANGLE = 30.0      # degrees - check this slice around our heading

# Wall following parameters (for going around obstacles)
WALL_SETPOINT = 0.30    # maintain this distance from wall (m)
WALL_KP = 1.5           # P-gain for wall following
WALL_SPEED = 0.30       # speed while following wall
WALL_DEADBAND = 0.02    # deadband around setpoint

# Control loop timing
LOOP_RATE = 20          # Hz
DT = 1.0 / LOOP_RATE

PRINT_DEBUG = True      # set to False to reduce console spam

# ==================== HELPER FUNCTIONS ====================

def clamp(x, min_val, max_val):
    """Keep x between min and max values"""
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    return x

def wrap_angle(angle):
    """Wrap angle to [-pi, pi] range"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_min_dist(ranges, thetas):
    """
    Find the minimum non-zero distance in the lidar scan.
    Returns (min_distance, angle_at_min)
    Ignores zero readings which are bad lidar returns.
    """
    if not ranges or not thetas:
        return float('inf'), 0.0
    
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    
    # only look at valid readings (non-zero)
    valid = r > 0.0
    
    if not np.any(valid):
        return float('inf'), 0.0
    
    r_valid = r[valid]
    t_valid = t[valid]
    
    idx = np.argmin(r_valid)
    return float(r_valid[idx]), float(t_valid[idx])

def find_min_ray_in_slice(ranges, thetas, target_angle, slice_deg=30.0):
    """
    Find minimum distance in a slice around the target angle.
    This tells us if there's an obstacle in the direction we want to go.
    
    target_angle: the direction we want to check (radians)
    slice_deg: how wide the slice should be (degrees)
    """
    if not ranges or not thetas:
        return float('inf')
    
    r = np.asarray(ranges, dtype=float)
    t = np.asarray(thetas, dtype=float)
    
    # wrap angles to [-pi, pi]
    t = np.arctan2(np.sin(t), np.cos(t))
    target_angle = wrap_angle(target_angle)
    
    # calculate angular difference
    angle_diff = np.abs(np.arctan2(np.sin(t - target_angle), 
                                    np.cos(t - target_angle)))
    
    half_slice = math.radians(slice_deg / 2.0)
    
    # find rays within the slice that are valid
    in_slice = (angle_diff <= half_slice) & (r > 0.0)
    
    if not np.any(in_slice):
        return float('inf')
    
    return float(np.min(r[in_slice]))

def cross_product_2d(v1, v2):
    """
    2D cross product returns a scalar (z-component).
    Used to determine left/right side.
    """
    return v1[0] * v2[1] - v1[1] * v2[0]

# ==================== DRIVE TO POSE ====================

def drive_to_pose(robot, goal_x, goal_y, goal_theta):
    """
    Drive the robot to a specific goal pose (x, y, theta).
    This is the "hit the spot" part - just drive straight to goal without obstacles.
    
    Returns True when goal is reached.
    """
    # get current position from odometry
    x, y, theta = robot.read_odometry()
    
    # calculate distance to goal
    dx = goal_x - x
    dy = goal_y - y
    dist = math.sqrt(dx**2 + dy**2)
    
    # if we're close enough to the goal position
    if dist < GOAL_XY_TOL:
        # now just fix the angle
        angle_error = wrap_angle(goal_theta - theta)
        
        if abs(angle_error) < GOAL_THETA_TOL:
            # we made it! stop the robot
            robot.drive(0, 0, 0)
            return True
        else:
            # rotate in place to match goal angle
            w = clamp(KP_ANGULAR * angle_error, -W_MAX, W_MAX)
            robot.drive(0, 0, w)
            return False
    
    # calculate angle to goal
    angle_to_goal = math.atan2(dy, dx)
    angle_error = wrap_angle(angle_to_goal - theta)
    
    # if we need to turn a lot, just rotate in place first
    if abs(angle_error) > math.radians(30):
        w = clamp(KP_ANGULAR * angle_error, -W_MAX, W_MAX)
        robot.drive(0, 0, w)
        return False
    
    # drive toward goal with proportional control
    v = clamp(KP_LINEAR * dist, V_MIN, V_MAX)
    w = clamp(KP_ANGULAR * angle_error, -W_MAX, W_MAX)
    
    robot.drive(v, 0, w)
    return False

# ==================== BUG NAVIGATION STATE MACHINE ====================

class BugNavigator:
    """
    State machine for bug navigation.
    States:
    - GO_TO_GOAL: driving directly toward goal
    - FOLLOW_WALL: following wall to get around obstacle
    - DONE: reached the goal
    """
    
    def __init__(self, robot):
        self.robot = robot
        self.state = "GO_TO_GOAL"
        
        # goal position
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_theta = 0.0
        
        # for wall following
        self.wall_follow_start_dist = None  # distance to goal when we started following wall
        self.leave_point_x = None           # where we left the wall
        self.leave_point_y = None
        
        # smoothing for distance measurements
        self.smooth_wall_dist = None
        self.smooth_alpha = 0.35
    
    def set_goal(self, x, y, theta):
        """Set the goal position for navigation"""
        self.goal_x = x
        self.goal_y = y
        self.goal_theta = theta
        self.state = "GO_TO_GOAL"
        
        if PRINT_DEBUG:
            print(f"\n[GOAL SET] Target: ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)")
    
    def update(self):
        """
        Main control loop - called repeatedly.
        Checks current state and executes appropriate behavior.
        """
        # get current pose
        x, y, theta = self.robot.read_odometry()
        
        # calculate distance to goal
        dist_to_goal = distance(x, y, self.goal_x, self.goal_y)
        angle_to_goal = math.atan2(self.goal_y - y, self.goal_x - x)
        
        # get lidar data
        try:
            ranges, thetas = self.robot.read_lidar()
        except:
            print("[WARN] Failed to read lidar")
            self.robot.drive(0, 0, 0)
            return
        
        # check if path to goal is clear
        # we need to check in the direction of the goal
        heading_to_goal = wrap_angle(angle_to_goal - theta)
        min_in_path = find_min_ray_in_slice(ranges, thetas, heading_to_goal, SLICE_ANGLE)
        
        # ===== STATE MACHINE =====
        
        if self.state == "GO_TO_GOAL":
            # check if we reached the goal
            if dist_to_goal < GOAL_XY_TOL:
                angle_error = abs(wrap_angle(self.goal_theta - theta))
                if angle_error < GOAL_THETA_TOL:
                    self.state = "DONE"
                    self.robot.drive(0, 0, 0)
                    print(f"\n[SUCCESS] Reached goal at ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)")
                    return
            
            # check for obstacles in our path
            if min_in_path < OBS_DETECT_DIST:
                # obstacle detected! switch to wall following
                self.state = "FOLLOW_WALL"
                self.wall_follow_start_dist = dist_to_goal
                self.leave_point_x = None
                self.leave_point_y = None
                self.smooth_wall_dist = None
                print(f"[STATE CHANGE] GO_TO_GOAL -> FOLLOW_WALL (obstacle at {min_in_path:.2f}m)")
            else:
                # path is clear, drive toward goal
                self._drive_to_goal(x, y, theta)
        
        elif self.state == "FOLLOW_WALL":
            # check if we can head back to goal
            # two conditions:
            # 1. path to goal is clear
            # 2. we're closer to goal than when we started following the wall
            
            path_clear = min_in_path > OBS_CLEAR_DIST
            closer_to_goal = dist_to_goal < (self.wall_follow_start_dist - 0.15)
            
            if path_clear and closer_to_goal:
                # we can leave the wall and head to goal
                self.state = "GO_TO_GOAL"
                self.leave_point_x = x
                self.leave_point_y = y
                print(f"[STATE CHANGE] FOLLOW_WALL -> GO_TO_GOAL (path clear, closer to goal)")
            else:
                # keep following the wall
                self._follow_wall(ranges, thetas)
        
        elif self.state == "DONE":
            # we're done, just stop
            self.robot.drive(0, 0, 0)
    
    def _drive_to_goal(self, x, y, theta):
        """
        Drive directly toward the goal.
        Uses proportional control for smooth motion.
        """
        dx = self.goal_x - x
        dy = self.goal_y - y
        dist = math.sqrt(dx**2 + dy**2)
        
        angle_to_goal = math.atan2(dy, dx)
        angle_error = wrap_angle(angle_to_goal - theta)
        
        # if we need to turn a lot, rotate in place
        if abs(angle_error) > math.radians(45):
            w = clamp(KP_ANGULAR * angle_error, -W_MAX, W_MAX)
            self.robot.drive(0, 0, w)
            if PRINT_DEBUG:
                print(f"[GO_TO_GOAL] Rotating to face goal | angle_err={math.degrees(angle_error):.1f}°")
            return
        
        # drive forward with course correction
        v = clamp(KP_LINEAR * dist, V_MIN, V_MAX)
        w = clamp(KP_ANGULAR * angle_error, -W_MAX, W_MAX)
        
        self.robot.drive(v, 0, w)
        
        if PRINT_DEBUG:
            print(f"[GO_TO_GOAL] dist={dist:.2f}m | v={v:.2f} | w={w:.2f}")
    
    def _follow_wall(self, ranges, thetas):
        """
        Follow the wall to get around an obstacle.
        Uses the same technique as the wall follower project.
        """
        # find nearest point (this is our wall)
        d_min, th_min = find_min_dist(ranges, thetas)
        
        if not math.isfinite(d_min):
            # no wall detected, just stop
            self.robot.drive(0, 0, 0)
            return
        
        # smooth the distance
        if self.smooth_wall_dist is None:
            self.smooth_wall_dist = d_min
        else:
            self.smooth_wall_dist = (self.smooth_alpha * d_min + 
                                    (1.0 - self.smooth_alpha) * self.smooth_wall_dist)
        
        # calculate wall normal (points from robot toward wall)
        nx = math.cos(th_min)
        ny = math.sin(th_min)
        
        # calculate tangent (perpendicular to normal)
        # tangent points along the wall
        tx = ny
        ty = -nx
        
        # prefer forward direction
        if tx < 0:
            tx = -tx
            ty = -ty
        
        # normalize
        mag = math.sqrt(tx**2 + ty**2)
        if mag > 0:
            tx /= mag
            ty /= mag
        
        # base velocity along wall
        vx_tangent = WALL_SPEED * tx
        vy_tangent = WALL_SPEED * ty
        
        # correction to maintain setpoint distance
        err = WALL_SETPOINT - self.smooth_wall_dist
        
        if abs(err) > WALL_DEADBAND:
            # need to correct distance
            # negative error means too close, move away from wall
            corr = -WALL_KP * err
        else:
            corr = 0.0
        
        # correction is in direction of normal
        vx_corr = corr * nx
        vy_corr = corr * ny
        
        # combine tangent and correction
        vx = vx_tangent + vx_corr
        vy = vy_tangent + vy_corr
        
        # limit speed
        speed = math.sqrt(vx**2 + vy**2)
        if speed > V_MAX:
            scale = V_MAX / speed
            vx *= scale
            vy *= scale
        
        # yaw alignment with tangent
        desired_yaw = math.atan2(ty, tx)
        w = clamp(1.0 * desired_yaw, -W_MAX, W_MAX)
        
        self.robot.drive(vx, vy, w)
        
        if PRINT_DEBUG:
            print(f"[FOLLOW_WALL] wall_dist={self.smooth_wall_dist:.2f}m | err={err:+.2f}m")

# ==================== MAIN PROGRAM ====================

def main():
    print("=" * 60)
    print("BUG NAVIGATION ALGORITHM")
    print("=" * 60)
    
    # connect to robot
    robot = MBot()
    print("[INFO] Connected to robot")
    
    # create navigator
    navigator = BugNavigator(robot)
    
    # get goal from user
    print("\nEnter goal position:")
    try:
        goal_x = float(input("  x (meters): "))
        goal_y = float(input("  y (meters): "))
        goal_theta_deg = float(input("  theta (degrees): "))
        goal_theta = math.radians(goal_theta_deg)
    except ValueError:
        print("[ERROR] Invalid input. Please enter numbers.")
        return
    
    # set the goal
    navigator.set_goal(goal_x, goal_y, goal_theta)
    
    print("\n[INFO] Starting bug navigation...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    try:
        # main control loop
        while navigator.state != "DONE":
            navigator.update()
            time.sleep(DT)
        
        # print final pose
        x, y, theta = robot.read_odometry()
        print(f"\n[FINAL POSE] x={x:.3f}m, y={y:.3f}m, theta={math.degrees(theta):.1f}°")
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.drive(0, 0, 0)
        print("[INFO] Robot stopped")

if __name__ == "__main__":
    main()
