from vehicle import Driver

# Initialize the driver (Altino uses the Car/Driver API)
driver = Driver()
timestep = int(driver.getBasicTimeStep())

# Enable keyboard
keyboard = driver.getKeyboard()
keyboard.enable(timestep)

# State variables
agent_mode = False
speed = 0.0
angle = 0.0

# Altino limits
MAX_SPEED = 1.8
MAX_ANGLE = 0.4

# Prevent the toggle from rapidly switching if you hold the key down for a few frames
toggle_pressed_last_frame = False

print("--- HYBRID ALTINO CONTROLLER STARTED ---", flush=True)
print("Press 'T' to toggle between Human and Agent control.", flush=True)
print("Use Arrow Keys to drive in Human mode.", flush=True)
print("Press 'Space' to brake.", flush=True)

while driver.step() != -1:
    
    # 1. GATHER ALL PRESSED KEYS
    # This loop grabs every key currently pressed so we can accelerate AND turn simultaneously
    keys = []
    k = keyboard.getKey()
    while k != -1:
        keys.append(k)
        k = keyboard.getKey()
        
    # 2. HANDLE MODE TOGGLING ('T' key)
    # Check for both uppercase and lowercase 't'
    if ord('t') in keys or ord('T') in keys:
        if not toggle_pressed_last_frame:
            agent_mode = not agent_mode
            print(f"\n[MODE SWITCH] -> {'AGENT' if agent_mode else 'HUMAN'} Control Active", flush=True)
            
            # Reset physical state so the car doesn't run away when switching
            speed = 0.0 
            angle = 0.0
            toggle_pressed_last_frame = True
    else:
        toggle_pressed_last_frame = False

    # 3. APPLY LOGIC BASED ON MODE
    if agent_mode:
        # ==========================================
        #               AGENT CONTROL
        # ==========================================
        print("RL AGENT CONTROL", flush=True)
        speed = 0.0
        angle = 0.0
        
    else:
        # ==========================================
        #               HUMAN CONTROL
        # ==========================================
        is_steering = False
        
        # Throttle
        if keyboard.UP in keys:
            speed += 0.05
        if keyboard.DOWN in keys:
            speed -= 0.05
            
        # Steering
        if keyboard.LEFT in keys:
            angle -= 0.05
            is_steering = True
        if keyboard.RIGHT in keys:
            angle += 0.05
            is_steering = True
            
        # Emergency Brake
        if ord(' ') in keys:
            speed = 0.0
            
        # Auto-center wheels if left/right are released
        if not is_steering:
            angle = 0.0

    # 4. ENFORCE PHYSICAL LIMITS
    if speed > MAX_SPEED: speed = MAX_SPEED
    if speed < -MAX_SPEED: speed = -MAX_SPEED
    if angle > MAX_ANGLE: angle = MAX_ANGLE
    if angle < -MAX_ANGLE: angle = -MAX_ANGLE

    # 5. SEND COMMANDS TO THE ALTINO
    driver.setCruisingSpeed(speed)
    driver.setSteeringAngle(angle)