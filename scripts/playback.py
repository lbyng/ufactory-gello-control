import socket
import time
import numpy as np
import json

def replay_recording(recording_path):
   try:
       print(f"Loading recording: {recording_path}")
       data = np.load(recording_path)
       positions = data['positions']
       quaternions = data['quaternions']
       grippers = data['grippers']
       timestamps = data['timestamps']
       
       client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       target_host = 'localhost'
       target_port = 12345
       
       try:
           client_socket.connect((target_host, target_port))
           print(f"Connected to {target_host}:{target_port}")
           
           start_time = time.time()
           print("Starting playback")
           data_length = len(positions)
           
           for i in range(data_length):
               next_time = start_time + timestamps[i]
               current_time = time.time()
               
               if current_time < next_time:
                   time.sleep(next_time - current_time)
               
               arm_data = {
                   'arm': {
                       'pos': positions[i].tolist(),
                       'quat': quaternions[i].tolist(),
                       'grip': float(grippers[i])
                   }
               }
               
               json_data = json.dumps(arm_data)
               client_socket.sendall(json_data.encode('utf-8'))
               
               if i == 0 or i == data_length - 1 or (i + 1) % 200 == 0:
                   print(f"Progress: {i+1}/{data_length}")
           
           print("Playback completed")
               
       except KeyboardInterrupt:
           print("\nInterrupted, stopping playback")
       finally:
           client_socket.close()
           print("Socket connection closed")
           
   except Exception as e:
       print(f"Playback error: {e}")

if __name__ == "__main__":
   import sys
   
   if len(sys.argv) != 2:
       print("Usage: python playback.py <recording_file_path>")
       sys.exit(1)
   
   recording_file = sys.argv[1]
   replay_recording(recording_file)