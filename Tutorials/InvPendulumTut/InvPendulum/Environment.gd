extends Node2D

var agent_action = [0.0]
var env_action = [0, 0]

var sem_action
var sem_observation
var sem_reset
var mem
onready var policy_data = load("res://ddpg_policy.tres")
var policy
var policy_action

var reset = false
var timeout = true
var deltat = 0.05
var time_elapsed = 0.0
var prev_time = 0.0
var sem_delta = 0.0
var target_delta = 0.025

var max_num_steps = 500
var num_steps = 0

func _ready():
	mem = cSharedMemory.new()
	if mem.exists():
		sem_action = cSharedMemorySemaphore.new()
		sem_observation = cSharedMemorySemaphore.new()
		sem_reset = cSharedMemorySemaphore.new()
		
		sem_action.init("sem_action")
		sem_observation.init("sem_observation")
		print("Running as OpenAIGym environment")
	else:
		#pass
		policy = cTorchModel.new()
		policy.set_data(policy_data)
		
	var v = $Anchor/PinJoint2D/RigidBody2D.transform.get_origin()
	var AnchorT = $Anchor.transform
	var JointT = $Anchor/PinJoint2D.transform
	$Anchor/PinJoint2D/RigidBody2D.init_anchor = AnchorT.xform(JointT.get_origin())
	$Anchor/PinJoint2D/RigidBody2D.init_origin = AnchorT.xform(JointT.xform(v))
	$Anchor/PinJoint2D/RigidBody2D.init_rotation = 0.0
	$Anchor/PinJoint2D/RigidBody2D.init_angular_velocity = 1.0
	$Anchor/PinJoint2D/RigidBody2D.reset = true
	set_physics_process(true)

func is_done():
	if num_steps>=max_num_steps:
		num_steps = 0
		return 1
	else:
		return 0
	
func _process(delta):
	if mem.exists():
		var cur_time = OS.get_ticks_usec()
		var fps_est = 1000000.0/(cur_time - prev_time - sem_delta)
		Engine.set_iterations_per_second(fps_est)
		Engine.set_time_scale(Engine.get_iterations_per_second()*target_delta)
		if sem_delta>0:
			print(fps_est, " | ",Engine.get_iterations_per_second(), " | ", sem_delta)
		sem_delta = 0.0
		prev_time = cur_time
	
func _physics_process(delta):
	if timeout:
		if mem.exists():
			var time_start = OS.get_ticks_usec()
			sem_action.wait()
			var time_end = OS.get_ticks_usec()
			sem_delta = time_end - time_start
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
			if policy_action != null:
				agent_action = policy_action
			agent_action[0]*=8.0
		
		$ActionLabel.text = "Action: "+str(agent_action)
		
		if env_action[0] == 1:
			$Anchor/PinJoint2D/RigidBody2D.reset = true
			time_elapsed = 0.0
			
		if env_action[1] == 1:
			get_tree().quit()
		
		$Anchor/PinJoint2D/RigidBody2D.torque = -agent_action[0]/1.0
		
		$Timer.start(deltat)
		timeout = false

func _on_Timer_timeout():
	var observation = $Anchor/PinJoint2D/RigidBody2D.get_observation()
	var reward = [$Anchor/PinJoint2D/RigidBody2D.get_reward()]
	$ObservationLabel.text = "Observation: "+str(observation)
	$RewardLabel.text = "Reward: "+str(reward)
	$TimeLabel.text = "Time:"+str(time_elapsed)
	if mem.exists():
		mem.sendFloatArray("observation", observation)
		mem.sendFloatArray("reward", reward)
		mem.sendIntArray("done", [is_done()])
		sem_observation.post()
	else:
		#pass
		policy_action = policy.run(observation)
	
	time_elapsed += deltat
	num_steps += 1
	timeout = true
