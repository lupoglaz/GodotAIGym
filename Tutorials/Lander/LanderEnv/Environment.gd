extends Node2D

var mem
var sem_action
var sem_observation
var sem_physics

var agent_action_tensor
var observation_tensor
var reward_tensor
var done_tensor
var env_action_tensor

var deltat = 0.01

var agent = null
var Agent = preload("res://Agent.tscn")
var ground = null
var Ground = preload("res://Ground.tscn")
var in_landing_area = false

var elapsed_time = 0.0
var max_time = 10.0

var prev_time = 0.0
var sem_delta = 0.0
var target_delta = 0.01

var timeout = true

onready var policy_data = load('res://ddpg_actor.tres')
var policy = null
var policy_action = null

func reset():
	if not(agent == null):
		remove_child(agent)
		agent.queue_free()
	agent = Agent.instance()
	agent.transform.origin = Vector2(rand_range(100,get_viewport().size.x - 100), 100.0)
	add_child(agent)
	if not(ground == null):
		remove_child(ground)
		ground.queue_free()
	ground = Ground.instance()
	ground.transform.origin = Vector2(get_viewport().size.x/2, get_viewport().size.y)
	add_child(ground)
	ground.connect("landing_body_entered", self, "_on_landing_body_entered")
	ground.connect("landing_body_exited", self, "_on_landing_body_exited")
	elapsed_time = 0.0

func _ready():
	mem = cSharedMemory.new()
	sem_physics = Semaphore.new()
	sem_physics.post()
	
	if mem.exists():
		sem_action = cSharedMemorySemaphore.new()
		sem_action.init("sem_action")
		sem_observation = cSharedMemorySemaphore.new()
		sem_observation.init("sem_observation")
		
		agent_action_tensor = mem.findFloatTensor("agent_action")
		observation_tensor = mem.findFloatTensor("observation")
		reward_tensor = mem.findFloatTensor("reward")
		done_tensor = mem.findIntTensor("done")
		env_action_tensor = mem.findIntTensor("env_action")
		print("Running as an OpenAI gym environment")
	else:
		policy = cTorchModel.new()
		policy.set_data(policy_data)
		
	reset()
	
func _process(delta):
	if mem.exists():
		var cur_time = OS.get_ticks_usec()
		var fps_est = 1000000.0/(cur_time - prev_time - sem_delta)
		if fps_est > 0:
			Engine.set_iterations_per_second(fps_est)
			Engine.set_time_scale(Engine.get_iterations_per_second()*target_delta)
		sem_delta = 0.0
		prev_time = cur_time

func _physics_process(delta):
	sem_physics.wait()
	if timeout:
		var env_action = [0, 0]
		var action = [0, 0]
		var observation = [0]
		if mem.exists():
			var time_start = OS.get_ticks_usec()
			sem_action.wait()
			var time_end = OS.get_ticks_usec()
			sem_delta = time_end - time_start
			action = agent_action_tensor.read()
			env_action = env_action_tensor.read()
		else:
			if Input.is_key_pressed(KEY_A):
				action[1] = -1
			if Input.is_key_pressed(KEY_D):
				action[1] = +1
			if Input.is_key_pressed(KEY_W):
				action[0] = +1
			if Input.is_key_pressed(KEY_ESCAPE):
				env_action[1] = 1
			if Input.is_key_pressed(KEY_ENTER):
				env_action[0] = 1
			if policy_action != null:
				action = policy_action
					
		if env_action[0] == 1:
			reset()
		if env_action[1] == 1:
			get_tree().quit()
		agent.act(action)
		$Timer.start(deltat)
		timeout = false
	sem_physics.post()

func done_and_reward(observation, deltat):
	var reward = 0.0
	var done = false
	if in_landing_area:
		reward += 10.0
		if observation[9]: #left leg
			reward += 25.0
			done = true
		if observation[10]: #right leg
			reward += 25.0
			done = true
		if observation[9] and observation[10]:
			reward += 40.0
	if agent.body_collided:
		reward -= 100.0
		done = true
	if observation[2] > 2.0:
		reward -= 20.0
		done = true
	
	reward += 1.0/(1.0 + observation[2])
	elapsed_time += deltat
	if elapsed_time > max_time:
		done = true
	return [reward, done]

func _on_Timer_timeout():
	sem_physics.wait()
	var observation = agent.get_observation(ground.area_center)
	var dandr = done_and_reward(observation, deltat)
	var reward = [dandr[0]]
	var done = [int(dandr[1])]
		
	if mem.exists():
#		env_action_tensor.wite([0,0])
		observation_tensor.write(observation)
		reward_tensor.write(reward)
		done_tensor.write(done)
		sem_observation.post()
	else:
		if policy != null:
			policy_action = policy.run(observation)
		
	$LabelObservation.text = ''
	$LabelObservation.text += str(observation) + '\n'
	$LabelObservation.text += str(in_landing_area) + '\n'
	$LabelObservation.text += str(reward) + '\n'
	$LabelObservation.text += str(done) + '\n'
	sem_physics.post()
	timeout = true
	
func _on_landing_body_entered(body):
	if body.get_instance_id() == agent.get_instance_id():
		in_landing_area = true

func _on_landing_body_exited(body):
	if body.get_instance_id() == agent.get_instance_id():
		in_landing_area = false
