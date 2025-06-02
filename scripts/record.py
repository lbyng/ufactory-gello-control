import socket
import threading
import time
import numpy as np
import os
import json
from datetime import datetime

SAVE_DIRECTORY = 'recordings'

if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

class GelloRecorder:
    def __init__(self):
        self.recording = False
        self.positions = []
        self.quaternions = []
        self.grippers = []
        self.timestamps = []
        self.start_time = None
        self.filename = None

    def start_recording(self):
        self.recording = True
        self.positions = []
        self.quaternions = []
        self.grippers = []
        self.timestamps = []
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{SAVE_DIRECTORY}/gello_recording_{timestamp}"
        print(f"Recording started, data will be saved to: {self.filename}.npz")

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        if len(self.positions) == 0:
            print("No data recorded, file not saved")
            return

        positions_array = np.array(self.positions)
        quaternions_array = np.array(self.quaternions)
        grippers_array = np.array(self.grippers)
        timestamps_array = np.array(self.timestamps)
        
        np.savez(self.filename, 
                 positions=positions_array,
                 quaternions=quaternions_array,
                 grippers=grippers_array,
                 timestamps=timestamps_array,
                 start_time=self.start_time)
        
        print(f"Recording ended, saved {len(self.positions)} data points to {self.filename}.npz")
        self.positions = []
        self.quaternions = []
        self.grippers = []
        self.timestamps = []

    def record_data(self, data_str):
        if not self.recording:
            return
            
        data = json.loads(data_str)
        
        if 'arm' in data:
            arm_data = data['arm']
            if 'pos' in arm_data and 'grip' in arm_data:
                position = np.array(arm_data['pos'])
                grip = float(arm_data['grip'])
                
                if 'quat' in arm_data:
                    quaternion = np.array(arm_data['quat'])
                else:
                    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                
                current_time = time.time() - self.start_time
                
                self.positions.append(position)
                self.quaternions.append(quaternion)
                self.grippers.append(grip)
                self.timestamps.append(current_time)
                
                if len(self.positions) % 100 == 0:
                    print(f"Recorded {len(self.positions)} data points...")

def monitor_gello_socket(recorder, host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Listening on {host}:{port} waiting for connection...")
    
    conn, addr = server_socket.accept()
    print(f"Accepted connection from {addr}")
    
    buffer = b''
    while True:
        data = conn.recv(4096)
        if not data:
            print("Connection closed")
            break
            
        buffer += data
        
        try:
            decoded = buffer.decode('utf-8')
            json_obj = json.loads(decoded)
            
            recorder.record_data(decoded)
            
            buffer = b''
        except json.JSONDecodeError:
            if '}' in decoded:
                parts = decoded.split('}', 1)
                if parts and len(parts) > 1:
                    first_json = parts[0] + '}'
                    try:
                        json_obj = json.loads(first_json)
                        recorder.record_data(first_json)
                        buffer = parts[1].encode('utf-8')
                    except:
                        pass
    
    conn.close()
    server_socket.close()
    print("Socket closed")

def main():
    recorder = GelloRecorder()
    
    monitor_thread = threading.Thread(target=monitor_gello_socket, args=(recorder, "localhost", 12345))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    print("Gello data recorder started")
    print("Commands:")
    print("  's' - Start recording")
    print("  'p' - Stop recording and save")
    print("  'q' - Quit program")
    
    try:
        while True:
            command = input("> ").strip().lower()
            
            if command == 's':
                recorder.start_recording()
            elif command == 'p':
                recorder.stop_recording()
            elif command == 'q':
                if recorder.recording:
                    recorder.stop_recording()
                break
            else:
                print("Unknown command, please enter 's', 'p' or 'q'")
                
    except KeyboardInterrupt:
        if recorder.recording:
            recorder.stop_recording()
    
    print("Program exited")

if __name__ == "__main__":
    import argparse
    main()