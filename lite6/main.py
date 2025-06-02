import os
import sys
import time
import socket
import json
import threading
import numpy as np
from xarm.wrapper import XArmAPI
import user_config as uc

received_data = {
    'pos': [400, 0, 150],
    'quat': [1, 0, 0, 0],
    'grip': 0.5
}
running = True


def socket_server():
    global received_data, running
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((uc.HOST, uc.PORT))
        server_socket.listen(1)
        print(f"Server listening on {uc.HOST}:{uc.PORT}")
        
        while running:
            try:
                server_socket.settimeout(1.0)
                client_socket, addr = server_socket.accept()
                print(f"Connected to client: {addr}")
                
                client_socket.settimeout(None)
                
                while running:
                    data = client_socket.recv(1024)
                    if not data:
                        print("Client disconnected")
                        break
                    
                    try:
                        json_data = json.loads(data.decode('utf-8'))
                        if 'arm' in json_data and 'pos' in json_data['arm']:
                            received_data['pos'] = json_data['arm']['pos']
                            
                            if 'quat' in json_data['arm']:
                                received_data['quat'] = json_data['arm']['quat']
                            
                            if 'grip' in json_data['arm']:
                                received_data['grip'] = json_data['arm']['grip']
                            
                            output = f"Received: pos={received_data['pos']}"
                            if 'quat' in json_data['arm']:
                                output += f", quat={received_data['quat']}"
                            if 'grip' in json_data['arm']:
                                output += f", grip={received_data['grip']}"
                            print(output)
                            
                    except json.JSONDecodeError:
                        print("Invalid JSON data received")
                    except Exception as e:
                        print(f"Error processing data: {e}")
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Connection error: {e}")
                time.sleep(1)
    
    except Exception as e:
        print(f"Server error: {e}")
    
    finally:
        server_socket.close()
        print("Socket server closed")


def map_axis_value(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        normalized = 0.5
    else:
        normalized = (value - src_min) / (src_max - src_min)
    
    mapped_value = dst_min + normalized * (dst_max - dst_min)
    mapped_value = max(dst_min, min(dst_max, mapped_value))
    
    return mapped_value

def map_position(controller_pos):
    x, y, z = controller_pos
    
    x_sim = map_axis_value(x, uc.X_MAP[0], uc.X_MAP[1], uc.X_MAP[2], uc.X_MAP[3])
    y_sim = map_axis_value(y, uc.Y_MAP[0], uc.Y_MAP[1], uc.Y_MAP[2], uc.Y_MAP[3])
    z_sim = map_axis_value(z, uc.Z_MAP[0], uc.Z_MAP[1], uc.Z_MAP[2], uc.Z_MAP[3])
    
    return [x_sim, y_sim, z_sim]

def map_grip_value(controller_min, controller_max, gripper_min, gripper_max, controller_value):
    if controller_max == controller_min:
        normalized = 0.5
    else:
        normalized = (controller_value - controller_min) / (controller_max - controller_min)
    
    mapped_value = gripper_min + (1 - normalized) * (gripper_max - gripper_min)
    mapped_value = max(gripper_min, min(gripper_max, mapped_value))
    
    return mapped_value

def map_gripper(controller_value):
    controller_min = -15
    controller_max = 1.75
    
    gripper_min = 0
    gripper_max = 850
    
    return map_grip_value(controller_min, controller_max, gripper_min, gripper_max, controller_value)


def map_rotation_quaternion(quat, arm_id="right"):
    """
    Maps input quaternion to a new coordinate frame for intuitive arm control.

    Args:
        quat (list): Original quaternion [w, x, y, z].
        arm_id (str): Arm identifier, either "left" or "right". Defaults to "right".

    Returns:
        list: Mapped quaternion [w, x, y, z].
    """

    # Convert quaternion to Euler angles (ZYX order)
    w, x, y, z = quat
    
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    # Coordinate frame adjustment
    original_roll = roll
    original_pitch = pitch
    original_yaw = yaw

    roll = -original_pitch # align x-axis forward
    pitch = original_roll # align y-axis right
    yaw = original_yaw + np.pi/2 # align z-axis down
    
    # Apply forward/backward tilt amplification
    if pitch > 0:
        pitch = pitch * 2
    else:
        pitch = pitch * 1.5

    if roll > 0:
        roll = roll * 1.2
    else:
        roll = roll * 1.0
    
    # Arm-specific remapping
    if arm_id == "left":
        mapped_roll = roll * 1.2
        mapped_pitch = -pitch
        mapped_yaw = yaw * 1.2
    else:
        mapped_roll = roll
        mapped_pitch = -pitch
        mapped_yaw = yaw
    
    # Convert adjusted Euler angles back to quaternion (ZYX order)
    cy = np.cos(mapped_yaw * 0.5)
    sy = np.sin(mapped_yaw * 0.5)
    cp = np.cos(mapped_pitch * 0.5)
    sp = np.sin(mapped_pitch * 0.5)
    cr = np.cos(mapped_roll * 0.5)
    sr = np.sin(mapped_roll * 0.5)
    
    mapped_w = cr * cp * cy + sr * sp * sy
    mapped_x = sr * cp * cy - cr * sp * sy
    mapped_y = cr * sp * cy + sr * cp * sy
    mapped_z = cr * cp * sy - sr * sp * cy
    
    return [mapped_w, mapped_x, mapped_y, mapped_z]


def quat_to_euler(quat):
    w, x, y, z = quat
    
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    
    return roll, pitch, yaw


def main():
    global received_data, running
    
    socket_thread = threading.Thread(target=socket_server)
    socket_thread.daemon = True
    socket_thread.start()
    
    arm = XArmAPI(uc.ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    
    arm.move_gohome(wait=True)
    print("Moved to home position")
    
    arm.set_position(x=400, y=0, z=150, roll=-180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)
    print("Moved to initial position")

    arm.set_mode(7)
    arm.set_state(0)
    time.sleep(1)
    print("Set to cartesian online trajectory planning mode")
    
    speed = 150
    gripper_last_value = -1
    
    try:
        print("Starting to receive controller data...")
        while True:
            current_pos = received_data['pos']
            current_quat = received_data['quat']
            current_grip = received_data['grip']
            
            mapped_pos = map_position(current_pos)
            x = mapped_pos[0]
            y = mapped_pos[1]
            z = mapped_pos[2]
            
            mapped_quat = map_rotation_quaternion(current_quat, arm_id="right")
            
            roll, pitch, yaw = quat_to_euler(mapped_quat)
            
            print(f"Mapped position: x={x:.1f}, y={y:.1f}, z={z:.1f}")
            print(f"Mapped orientation: roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")
            
            arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                             speed=speed, is_radian=False, wait=False)
            
            if abs(current_grip - gripper_last_value) > 0.05:

                if current_grip < 0:
                    try:
                        arm.set_vacuum_gripper(True, hardware_version=1)
                        print("Vacuum Gripper: ON")
                    except:
                        print("Failed to turn on vacuum gripper")
                else:
                    try:
                        arm.set_vacuum_gripper(False, hardware_version=1)
                        print("Vacuum Gripper: OFF")
                    except:
                        print("Failed to turn off vacuum gripper")
                gripper_last_value = current_grip
            
            time.sleep(0.01)  # 100Hz update rate
            
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        running = False
        print("Stopping arm and resetting...")
        
        arm.set_mode(0)
        arm.set_state(0)
        
        try:
            arm.move_gohome(wait=True)
        except:
            print("Failed to move home, emergency stopping")
        
        arm.disconnect()
        print("Arm disconnected")
        
        socket_thread.join(timeout=2.0)
        print("Program terminated")


if __name__ == "__main__":
    main()