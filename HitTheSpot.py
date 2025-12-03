import time
import math
from mbot_bridge.api import MBot

# how close is "close enough" to the goal
POSITION_TOLERANCE = 0.08   # meters - need to be within 8cm of goal x,y
ANGLE_TOLERANCE = 0.12      # radians - need to be within ~7 degrees of goal angle

# control gains - tune these if the robot is too aggressive or too slow
KP_DISTANCE = 0.9           # proportional gain for forward/backward speed
KP_ANGLE = 1.8              # proportional gain for turning

# speed limits
MAX_LINEAR_SPEED = 0.45     # max speed moving forward/backward (m/s)
MIN_LINEAR_SPEED = 0.10     # minimum speed to actually move (overcome friction)
MAX_ANGULAR_SPEED = 1.3     # max turning speed (rad/s)

# control loop rate
LOOP_HZ = 20
DT = 1.0 / LOOP_HZ

# ==================== HELPER FUNCTIONS ====================

def clamp(value, min_val, max_val):
    """Keep value between min and max"""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value

def normalize_angle(angle):
    """
    Wrap angle to be between -pi and pi.
    This handles the wraparound at +/- 180 degrees.
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def calculate_distance(x1, y1, x2, y2):
    """Simple Euclidean distance between two points"""
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx*dx + dy*dy)

# ==================== DRIVE TO POSE FUNCTION ====================

def drive_to_pose(robot, goal_x, goal_y, goal_theta, verbose=True):
    """
    Drive the robot to a specific goal pose.
    
    Parameters:
    - robot: MBot object
    - goal_x: target x position (meters)
    - goal_y: target y position (meters)
    - goal_theta: target angle (radians)
    - verbose: whether to print debug info
    
    Returns: True when goal is reached
    """
    
    # read where we are right now
    current_x, current_y, current_theta = robot.read_odometry()
    
    # calculate how far we are from the goal
    dx = goal_x - current_x
    dy = goal_y - current_y
    distance_to_goal = math.sqrt(dx*dx + dy*dy)
    
    # --- PHASE 1: Get to the right position (x, y) ---
    
    if distance_to_goal > POSITION_TOLERANCE:
        # we're not at the goal position yet, need to drive there
        
        # calculate what angle we need to face to drive toward goal
        desired_heading = math.atan2(dy, dx)
        
        # calculate how much we need to turn
        heading_error = normalize_angle(desired_heading - current_theta)
        
        # if we're facing way wrong, just spin in place first
        if abs(heading_error) > math.radians(40):
            # pure rotation - don't move forward yet
            angular_velocity = clamp(KP_ANGLE * heading_error, 
                                    -MAX_ANGULAR_SPEED, 
                                    MAX_ANGULAR_SPEED)
            
            robot.drive(0, 0, angular_velocity)
            
            if verbose:
                print(f"  Rotating to face goal | heading error: {math.degrees(heading_error):.1f}°")
            
            return False
        
        # we're facing roughly the right way, now drive forward while correcting heading
        
        # linear velocity based on how far we need to go
        linear_velocity = KP_DISTANCE * distance_to_goal
        
        # clamp it and enforce minimum
        linear_velocity = clamp(linear_velocity, 0, MAX_LINEAR_SPEED)
        if linear_velocity > 0:
            linear_velocity = max(linear_velocity, MIN_LINEAR_SPEED)
        
        # angular velocity to correct heading as we drive
        angular_velocity = clamp(KP_ANGLE * heading_error,
                                -MAX_ANGULAR_SPEED,
                                MAX_ANGULAR_SPEED)
        
        robot.drive(linear_velocity, 0, angular_velocity)
        
        if verbose:
            print(f"  Driving to goal | dist: {distance_to_goal:.3f}m | heading err: {math.degrees(heading_error):.1f}°")
        
        return False
    
    # --- PHASE 2: Fix the final angle ---
    
    # we're at the right position, now just need to face the right direction
    angle_error = normalize_angle(goal_theta - current_theta)
    
    if abs(angle_error) > ANGLE_TOLERANCE:
        # need to rotate to match goal angle
        angular_velocity = clamp(KP_ANGLE * angle_error,
                                -MAX_ANGULAR_SPEED,
                                MAX_ANGULAR_SPEED)
        
        robot.drive(0, 0, angular_velocity)
        
        if verbose:
            print(f"  Adjusting final angle | error: {math.degrees(angle_error):.1f}°")
        
        return False
    
    # --- SUCCESS: We're at the goal! ---
    
    robot.drive(0, 0, 0)  # stop the robot
    
    if verbose:
        print(f"  ✓ Reached goal!")
        print(f"    Final pose: ({current_x:.3f}, {current_y:.3f}, {math.degrees(current_theta):.1f}°)")
    
    return True

# ==================== MAIN PROGRAM ====================

def main():
    print("=" * 60)
    print("HIT THE SPOT - Drive to Goal Pose")
    print("=" * 60)
    
    # connect to the robot
    robot = MBot()
    print("\n[INFO] Connected to robot")
    
    # read starting position
    start_x, start_y, start_theta = robot.read_odometry()
    print(f"[INFO] Starting pose: ({start_x:.3f}, {start_y:.3f}, {math.degrees(start_theta):.1f}°)")
    
    # get goal from user
    print("\nEnter goal pose:")
    try:
        goal_x = float(input("  Goal X (meters): "))
        goal_y = float(input("  Goal Y (meters): "))
        goal_theta_deg = float(input("  Goal Theta (degrees): "))
        goal_theta = math.radians(goal_theta_deg)
    except ValueError:
        print("[ERROR] Invalid input. Please enter numbers only.")
        return
    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user")
        return
    
    print(f"\n[GOAL] Driving to ({goal_x:.2f}, {goal_y:.2f}, {goal_theta_deg:.1f}°)")
    print("[INFO] Press Ctrl+C to stop\n")
    
    try:
        # keep running the control loop until we reach the goal
        reached = False
        while not reached:
            reached = drive_to_pose(robot, goal_x, goal_y, goal_theta, verbose=True)
            time.sleep(DT)
        
        # print final confirmation
        final_x, final_y, final_theta = robot.read_odometry()
        print(f"\n{'='*60}")
        print("SUCCESS - Goal Reached!")
        print(f"{'='*60}")
        print(f"Final Position: ({final_x:.3f}, {final_y:.3f}, {math.degrees(final_theta):.1f}°)")
        
        # calculate final error
        final_dist = calculate_distance(final_x, final_y, goal_x, goal_y)
        final_angle_err = abs(normalize_angle(goal_theta - final_theta))
        print(f"Position Error: {final_dist*100:.2f} cm")
        print(f"Angle Error: {math.degrees(final_angle_err):.2f}°")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Something went wrong: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # always stop the robot when we're done
        robot.drive(0, 0, 0)
        print("[INFO] Robot stopped safely")

if __name__ == "__main__":
    main()
