# ruff: noqa
import time
from typing import List, Optional
import socket
import json
import mujoco
import mujoco.viewer
import numpy as np
from STservo_sdk import *
import user_config as uc

DEFAULT_NORMALIZATION_LIMITS = [
    [481, 3696],
    [1151, 2706],
    [2330, 812],
    [767, 3295],
    [1533, 3623],
    [919, 3528],
    [2125, 1881],
]

DEFAULT_JOINT_LIMITS = [
    [-235, 35],
    [0, 135],
    [-135, 0],
    [-202.5, 22.5],
    [-90, 90],
    [-202.5, 22.5],
    [180, -180],
]

def euler_to_quat(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def rmat_to_quat(rmat):
    trace = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (rmat[2, 1] - rmat[1, 2]) / S
        y = (rmat[0, 2] - rmat[2, 0]) / S
        z = (rmat[1, 0] - rmat[0, 1]) / S
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        S = np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2]) * 2
        w = (rmat[2, 1] - rmat[1, 2]) / S
        x = 0.25 * S
        y = (rmat[0, 1] + rmat[1, 0]) / S
        z = (rmat[0, 2] + rmat[2, 0]) / S
    elif rmat[1, 1] > rmat[2, 2]:
        S = np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2]) * 2
        w = (rmat[0, 2] - rmat[2, 0]) / S
        x = (rmat[0, 1] + rmat[1, 0]) / S
        y = 0.25 * S
        z = (rmat[1, 2] + rmat[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1]) * 2
        w = (rmat[1, 0] - rmat[0, 1]) / S
        x = (rmat[0, 2] + rmat[2, 0]) / S
        y = (rmat[1, 2] + rmat[2, 1]) / S
        z = 0.25 * S
    
    return np.array([w, x, y, z])

class GelloController:
    def __init__(
        self,
        device_name: str,
        arm_id: str = "arm",
        baud_rate: int = 1000000,
    ):
        self.baud_rate = baud_rate
        self.device_name = device_name
        self.arm_id = arm_id
        self.num_motors = 7
        self.motor_pos = np.zeros(self.num_motors, dtype=np.float32)
        self.motor_speed = np.zeros(self.num_motors, dtype=np.float32)
        
        self.portHandler = PortHandler(self.device_name)
        self.packetHandler = sts(self.portHandler)

        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open {arm_id} port: {device_name}")
        
        print(f"Successfully opened {arm_id} port: {device_name}")

        if not self.portHandler.setBaudRate(self.baud_rate):
            self.portHandler.closePort()
            raise RuntimeError(f"{arm_id} baud rate setting failed")
        
        print(f"{arm_id} baud rate set successfully")

        self.groupSyncRead = GroupSyncRead(self.packetHandler, STS_PRESENT_POSITION_L, 4)

    def read_joints(self):
        for sts_id in range(1, self.num_motors + 1):
            sts_addparam_result = self.groupSyncRead.addParam(sts_id)
            if not sts_addparam_result:
                raise RuntimeError(f"[{self.arm_id}][ID:{sts_id:03d}] groupSyncRead addparam Failed")

        sts_comm_result = self.groupSyncRead.txRxPacket()
        if sts_comm_result != COMM_SUCCESS:
            raise RuntimeError(f"[{self.arm_id}]{self.packetHandler.getTxRxResult(sts_comm_result)}")

        for sts_id in range(1, 8):
            sts_data_result, sts_error = self.groupSyncRead.isAvailable(sts_id, STS_PRESENT_POSITION_L, 4)
            if not sts_data_result:
                raise RuntimeError(f"[{self.arm_id}][ID:{sts_id:03d}] groupSyncRead getdata Failed")
            
            if sts_error != 0:
                raise RuntimeError(f"[{self.arm_id}]{self.packetHandler.getRxPacketError(sts_error)}")
            
            sts_present_position = self.groupSyncRead.getData(sts_id, STS_PRESENT_POSITION_L, 2)
            sts_present_speed = self.groupSyncRead.getData(sts_id, STS_PRESENT_SPEED_L, 2)
            self.motor_pos[sts_id - 1] = sts_present_position
            self.motor_speed[sts_id - 1] = self.packetHandler.sts_tohost(sts_present_speed, 15)
            
        self.groupSyncRead.clearParam()
        return self.motor_pos, self.motor_speed

    def close(self):
        print(f"Close {self.arm_id} Port")
        self.portHandler.closePort()

    def __del__(self):
        self.close()


class GelloControllerWrapper:
    def __init__(
        self,
        controller,
        motor_limits=None,
        joint_limits=None,
        use_mujoco=True,
        xml_path="gamify/controllers/gello_arm.xml",
    ):
        super().__init__()

        self.controller = controller
        self.motor_limits = motor_limits
        self.joint_limits = joint_limits
        if self.motor_limits is None:
            self.motor_limits = DEFAULT_NORMALIZATION_LIMITS
        if self.joint_limits is None:
            self.joint_limits = DEFAULT_JOINT_LIMITS

        self.motor_limits = np.array(self.motor_limits)
        self.joint_limits = np.radians(self.joint_limits)

        if use_mujoco:
            self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mujoco_data = mujoco.MjData(self.mujoco_model)

    def get_joint_pos(self):
        pos, _ = self.controller.read_joints()
        pos = (pos - self.motor_limits[:, 1]) / (self.motor_limits[:, 0] - self.motor_limits[:, 1])
        pos = self.joint_limits[:, 0] + pos * (self.joint_limits[:, 1] - self.joint_limits[:, 0])
        return pos

    def close(self):
        self.controller.close()

    def get_ee_pos_rmat_gripper(self):
        qpos = self.get_joint_pos()
        
        self.mujoco_data.qpos[:] = qpos[:-1]
        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        link6_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        pos = np.array(self.mujoco_data.xpos[link6_id])
        rmat = np.array(self.mujoco_data.xmat[link6_id]).reshape(3, 3)

        pos = np.array([pos[1], pos[0], pos[2]])
        rmat = np.array([rmat[1], -rmat[0], rmat[2]])
        
        gripper = qpos[-1]

        return pos, rmat, gripper

class SingleArmClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.connected = True
        print(f"Connected to simulator: {self.host}:{self.port}")
        return True
    
    def send_data(self, pos=None, quat=None, grip=None):
        if not self.connected:
            self.connect()
                
        if pos is not None and grip is not None:
            data = {
                'arm': {
                    'pos': pos.tolist() if isinstance(pos, np.ndarray) else pos,
                    'grip': float(grip)
                }
            }
            
            if quat is not None:
                data['arm']['quat'] = quat.tolist() if isinstance(quat, np.ndarray) else quat
            
            json_data = json.dumps(data)
            self.socket.sendall(json_data.encode('utf-8'))
            return True
        return False
            
    def close(self):
        if self.connected and self.socket:
            self.socket.close()
            print("Client connection closed")
            self.connected = False

if __name__ == "__main__":
    
    # Create objects
    controller = GelloController(device_name=uc.DEVICE_NAME, arm_id="arm", baud_rate=1000000)
    wrapper = GelloControllerWrapper(controller=controller)
    client = SingleArmClient(host=uc.HOST, port=uc.PORT)
    
    try:
        if uc.USE_MUJOCO:
            view_model = wrapper.mujoco_model
            view_data = wrapper.mujoco_data
            
            with mujoco.viewer.launch_passive(view_model, view_data) as viewer:
                print("Starting to send controller data to simulator...")
                
                while True:
                    pos, rmat, gripper = wrapper.get_ee_pos_rmat_gripper()
                    quat = rmat_to_quat(rmat)
                    client.send_data(pos, quat, gripper)
                    print(f"Arm: Position {pos}, Quat {quat}, Gripper {gripper:.4f}")
                    time.sleep(0.02)
                    viewer.sync()
        else:
            print("Starting to send controller data to simulator...")
            
            last_print_time = 0
            
            while True:
                pos, rmat, gripper = wrapper.get_ee_pos_rmat_gripper()
                quat = rmat_to_quat(rmat)
                client.send_data(pos, quat, gripper)
                current_time = time.time()
                if current_time - last_print_time >= 2.0:
                    print("-" * 50)
                    print(f"Arm: Position {pos}")
                    if quat is not None:
                        print(f"Quat: {quat}")
                    print(f"Gripper: {gripper:.4f}")
                    last_print_time = current_time
                
                time.sleep(0.02)
                
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        client.close()
        wrapper.close()