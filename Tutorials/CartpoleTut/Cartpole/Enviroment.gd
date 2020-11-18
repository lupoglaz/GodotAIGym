extends Node2D

var agent_action = [0.0]
var env_action = [0, 0]

var sem_action
var sem_observation
var sem_reset
var mem

var reset = false
var timeout = true
var time_elapsed = 0.0
var steps_beyond_done = -1


# Called when the node enters the scene tree for the first time.
func _ready():
	$Player/whole/PinJoint2D/stick.tau = $Timer.wait_time
	$Player/whole.init_transform = $Player.transform
	$Player/whole/PinJoint2D/stick.init_transform = $Player.transform
	
	var v = $Player/whole/PinJoint2D/stick.transform.get_origin()
	var AnchorT = $Player/whole.transform
	var JointT = $Player/whole/PinJoint2D.transform
	$Player/whole/PinJoint2D/stick.init_anchor = AnchorT.xform(JointT.get_origin())
	$Player/whole/PinJoint2D/stick.init_origin = AnchorT.xform(JointT.xform(v))
	$Player/whole/PinJoint2D/stick.init_rotation = 0.0
	
	time_elapsed = 0.0
	set_physics_process(true)

	
	
func _physics_process(delta):
	
	
	if timeout:
		agent_action[0] = 0.0
		env_action[0] = 0
		env_action[1] = 0
		if Input.is_action_pressed("ui_right"):
			agent_action[0] = 1.0
		if Input.is_action_pressed("ui_left"):
			agent_action[0] = -1.0
		if Input.is_key_pressed(KEY_ENTER):
			env_action[0] = 1
		if Input.is_key_pressed(KEY_ESCAPE):
			env_action[1] = 1
		
		if env_action[0] == 1:
			time_elapsed = 0.0
			$Player/whole.reset = true
			
		if env_action[1] == 1:
			get_tree().quit()
			
			
		$Timer.start()
		timeout = false
	
	
func _on_Timer_timeout():
	time_elapsed += $Timer.wait_time
	var state = $Player/whole.get_state()
	if state != null:
		var reward
		
		var theta_threshold_radians = 12 * 2 * PI / 360
		
		var done = (state[0] < -2.4) or  (state[1] > 2.4)
		done = done or state[2] < -theta_threshold_radians
		done = done or  state[2] > theta_threshold_radians
	 
		if not done:
			reward = 1
		elif steps_beyond_done == -1:
			reward = -100
			steps_beyond_done = 0
		else:
			#print("Error!! Steps Beyond done")
			steps_beyond_done += 1
			reward = 0
			
		#print(done)
		$Reward.text = "Reward: " + str(reward)
		$State.text = "State: " + str(state)
	
	
	
	
	
	timeout=true
