extends Node2D

var sem_action
var sem_observation
var sem_reset
var mem

var agent_action = [0.0]
var env_action = [0, 0]

var timeout = true
var time_elapsed = 0.0

var done = false
var previous_state


# Called when the node enters the scene tree for the first time.
func _ready():
	mem = cSharedMemory.new()
	if mem.exists():
		sem_action = cSharedMemorySemaphore.new()
		sem_observation = cSharedMemorySemaphore.new()
		sem_reset = cSharedMemorySemaphore.new()
		
		sem_action.init("sem_action")
		sem_observation.init("sem_observation")
		print("Running as OpenAIGym environment")
	set_physics_process(true)


	
func _process(delta):
	
	if timeout:
#		Engine.iterations_per_second = max(60, Engine.get_frames_per_second())
#		Engine.time_scale = min(1.0, Engine.iterations_per_second/10.0)
		
		if mem.exists():
			sem_action.wait()
			agent_action = mem.getFloatArray("agent_action")
			env_action = mem.getIntArray("env_action")
		else:
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
			done = false
			
		if env_action[1] == 1:
			get_tree().quit()
		
		$Timer.start()
		timeout = false

func calculate_reward(state, next_state):
	var screen_size = get_viewport().size.x
	#We scale the variables
	if done:
		return 100
	if abs(next_state[2]) > 0.5:
		# The car is upside down
		return -100
	var reward
	var potencial_e1 = (get_viewport().size.x - state[0] - (get_viewport().size.x - $Ground.maxi_car.y))
	var potencial_e2 = (get_viewport().size.x - next_state[0] - (get_viewport().size.x - $Ground.maxi_car.y))
	
	var kinetic1 = state[1]*state[1]
	var kinetic2 = next_state[1]*next_state[1]
	reward = (potencial_e2 + kinetic2) - (potencial_e1 + kinetic1)
	reward /=2000
	
	return reward

func normalize_state(state):
	var one = (get_viewport().size.x - state[0] - (get_viewport().size.x - $Ground.maxi_car.y))
	one /= 200
	return [one, state[1]/400, state[2]]

func _on_Timer_timeout():
	var state = $Car.get_state()
	var reward = 0
	if previous_state != null:
		reward = calculate_reward(previous_state, state)
		#reward *=100
		previous_state = state
	else:
		previous_state = state
	
	
	
	
	
	
	$State.text = "State: " + str(normalize_state(state))
	$Reward.text = "Reward: " + str(reward)
	
	if mem.exists():
		mem.sendFloatArray("observation", normalize_state(state))
		mem.sendFloatArray("reward", [reward])
		mem.sendIntArray("done", [done])
		sem_observation.post()
	
	time_elapsed+=$Timer.wait_time
	timeout = true


func _on_Goal_area_entered(area):
	pass


func _on_Goal_body_entered(body):
	done = true
