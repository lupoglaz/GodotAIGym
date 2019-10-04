extends Node2D
var agent_action = [0.0, 0.0]
var env_action = [0, 0]
var time_elapsed = 0.0
var timeout = true
var deltat = 0.01

var status_in_landing = false

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
		if agent_action[1] > 0.5:
			$Lander/LeftEngine.emitting = true
		if agent_action[1] < -0.5:
			$Lander/RightEngine.emitting = true
		
		$Timer.start(deltat)
		timeout = false

func _on_Timer_timeout():
	time_elapsed += deltat
	timeout = true
