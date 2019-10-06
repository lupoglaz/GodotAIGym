extends Node2D
var agent_action = [0.0, 0.0]
var env_action = [0, 0]
var time_elapsed = 0.0
var timeout = true
var deltat = 0.01

var in_landing_area = false

var prev_shaping
var FPS = 60.0

func _ready():
	set_physics_process(true)


func _process(delta):
	if timeout:
		agent_action[0] = 0.0
		agent_action[1] = 0.0
		if Input.is_key_pressed(KEY_SPACE):
			agent_action[0] = 1.0
		if Input.is_key_pressed(KEY_A):
			agent_action[1] = 1.0
		if Input.is_key_pressed(KEY_D):
			agent_action[1] = -1.0
		env_action[0] = 0
		env_action[1] = 0
		if Input.is_key_pressed(KEY_ENTER):
			env_action[0] = 1
		if Input.is_key_pressed(KEY_ESCAPE):
			env_action[1] = 1
			
			
		if env_action[0] == 1:
			$Lander.reset = true
			$Ground.reset = true
			time_elapsed = 0.0
			
		if env_action[1] == 1:
			get_tree().quit()
		
		$Lander/MainEngine.emitting = false
		$Lander/LeftEngine.emitting = false
		$Lander/RightEngine.emitting = false
		if agent_action[0] > 0.5:
			$Lander/MainEngine.emitting = true
			$Lander.main_engine_power = abs(agent_action[0])
		if agent_action[1] > 0.5:
			$Lander/LeftEngine.emitting = true
			$Lander.left_engine_power = abs(agent_action[1])
		if agent_action[1] < -0.5:
			$Lander/RightEngine.emitting = true
			$Lander.right_engine_power = abs(agent_action[1])
		
		$Timer.start(deltat)
		timeout = false
		
func get_state():
	var SCALE_X = float(get_viewport().size.x*0.5)
	var SCALE_Y = float(get_viewport().size.y*0.5)	
	var state = [($Lander.pos_x - $Ground.area_center.x)/SCALE_X,
		($Lander.pos_y - $Ground.area_center.y)/SCALE_Y,
		$Lander.vel_x/SCALE_X,
		$Lander.vel_y/SCALE_Y,
		$Lander.angle,
		20.0*$Lander.angular_vel/FPS,
		0,
		0
		]
	if $Lander.left_leg_collided:
		state[6] = 1.0
	if $Lander.right_leg_collided:
		state[7] = 1.0
	return state

func _on_Timer_timeout():
	time_elapsed += deltat
	timeout = true
	
	var reward = 0.0
	var done = false
	var state = get_state()
	
	var shaping = -100*sqrt(state[0]*state[0] + state[1]*state[1]) - 100*sqrt(state[2]*state[2] + state[3]*state[3]) - 100*abs(state[4]) + 10*state[6] + 10*state[7]
	if not(prev_shaping == null):
		reward = shaping - prev_shaping
	prev_shaping = shaping
	
	if $Lander.body_collided or abs(state[0])>1.0 or abs(state[1])>1.0:
		reward -= 100.0
		done = true
	if $Lander/MainEngine.emitting:
		reward -= 0.3
	if $Lander/LeftEngine.emitting or $Lander/RightEngine.emitting: 
		reward -= 0.03
	if $Lander.rest:
		reward += 100.0
		done = true
		
	$RewardLabel.text = "Reward:"+str(reward)
	$StateLabel.text = "State:"+str(state)
	

func _on_LandingArea_body_entered(body):
	if body.get_instance_id() == $Lander.get_instance_id():
		in_landing_area = true
	
func _on_LandingArea_body_exited(body):
	if body.get_instance_id() == $Lander.get_instance_id():
		in_landing_area = false
	