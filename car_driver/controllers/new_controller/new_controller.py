# from vehicle import Driver

# #Initialize the driver
# driver = Driver()
# timestep = int(driver.getBasicTimeStep())

# #Enable keyboard
# keyboard = driver.getKeyboard()
# keyboard.enable(timestep)

# #State variables
# agent_mode = False
# speed = 0.0
# angle = 0.0

# #Altino limits
# MAX_SPEED = 1.8
# MAX_ANGLE = 0.4

# toggle_pressed_last_frame = False

# print("--- HYBRID ALTINO CONTROLLER STARTED ---", flush=True)
# print("Press 'T' to toggle between Human and Agent control.", flush=True)
# print("Use Arrow Keys to drive in Human mode.", flush=True)
# print("Press 'Space' to brake.", flush=True)

# while driver.step() != -1:
    
    # #1. GATHER ALL PRESSED KEYS
    # keys = []
    # k = keyboard.getKey()
    # while k != -1:
        # keys.append(k)
        # k = keyboard.getKey()
        
    # #2. HANDLE MODE TOGGLING ('T' key)
    # if ord('t') in keys or ord('T') in keys:
        # if not toggle_pressed_last_frame:
            # agent_mode = not agent_mode
            # print(f"\n[MODE SWITCH] -> {'AGENT' if agent_mode else 'HUMAN'} Control Active", flush=True)
            # speed = 0.0 
            # angle = 0.0
            # toggle_pressed_last_frame = True
    # else:
        # toggle_pressed_last_frame = False

    # #3. APPLY LOGIC BASED ON MODE
    # if agent_mode:
        # Agent mode idles safely for now to avoid the getField() crash
        # speed = 0.0
        # angle = 0.0
        
    # else:
        # is_steering = False
        
        # #Throttle
        # if keyboard.UP in keys: speed += 0.05
        # if keyboard.DOWN in keys: speed -= 0.05
            
        # #Steering
        # if keyboard.LEFT in keys:
            # angle -= 0.05
            # is_steering = True
        # if keyboard.RIGHT in keys:
            # angle += 0.05
            # is_steering = True
            
        # #Emergency Brake
        # if ord(' ') in keys: speed = 0.0
            
        # #Auto-center
        # if not is_steering: angle = 0.0

    # #4. ENFORCE PHYSICAL LIMITS
    # if speed > MAX_SPEED: speed = MAX_SPEED
    # if speed < -MAX_SPEED: speed = -MAX_SPEED
    # if angle > MAX_ANGLE: angle = MAX_ANGLE
    # if angle < -MAX_ANGLE: angle = -MAX_ANGLE

    # #5. SEND COMMANDS TO THE ALTINO
    # driver.setCruisingSpeed(speed)
    # driver.setSteeringAngle(angle)
    
    
from vehicle import Driver
import sys

print("--- [ACTUATOR] LAUNCHING VEHICLE CONTROLLER ---", flush=True)

try:
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())
    
    # Initialize Altino's built-in Receiver
    receiver = driver.getDevice("receiver")
    receiver.enable(timestep)
    
    print("--- [ACTUATOR] INITIALIZED SUCCESSFULLY WITH RECEIVER ---", flush=True)
except Exception as e:
    print(f"--- [ACTUATOR] INITIALIZATION ERROR: {e} ---", flush=True)
    sys.exit(1)

while driver.step() != -1:
    # Check if a training step command packet is available
    if receiver.getQueueLength() > 0:
        # Webots automatically converts the data to a Python string
        data = receiver.getString()
        receiver.nextPacket()  # Remove packet from the queue
        
        try:
            speed_str, angle_str = data.split(",")
            speed = float(speed_str)
            angle = float(angle_str)
            
            # Apply physics commands to Altino
            driver.setCruisingSpeed(speed)
            driver.setSteeringAngle(angle)
        except Exception as e:
            print(f"[Actuator] Error parsing data: {e}", flush=True)
 
 
 
 