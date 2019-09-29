extends Node2D

var agent_action = [0.0]
var env_action = [0, 0]

var sem_action
var sem_observation
var sem_reset
var mem

var reset = false
var timeout = true
var deltat = 0.05
var time_elapsed = 0.0

func _ready():
	if OS.get_name()=='X11_SHARED':
		sem_action = cSharedMemorySemaphore.new()
		sem_observation = cSharedMemorySemaphore.new()
		sem_reset = cSharedMemorySemaphore.new()
		mem = cSharedMemory.new()
		sem_action.init("sem_action")
		sem_observation.init("sem_observation")
		print("Running as OpenAIGym environment")
	
	var v = $Anchor/PinJoint2D/RigidBody2D.transform.get_origin()
	var AnchorT = $Anchor.transform
	var JointT = $Anchor/PinJoint2D.transform
	$Anchor/PinJoint2D/RigidBody2D.init_anchor = AnchorT.xform(JointT.get_origin())
	$Anchor/PinJoint2D/RigidBody2D.init_origin = AnchorT.xform(JointT.xform(v))
	$Anchor/PinJoint2D/RigidBody2D.init_rotation = 0.0
	$Anchor/PinJoint2D/RigidBody2D.init_angular_velocity = 0.0
	$Anchor/PinJoint2D/RigidBody2D.init_linear_velicity = Vector2(0.0, 0.0)
	
	set_physics_process(true)

func is_done():
	return 0
	
func _physics_process(delta):
	
	if timeout:
		Engine.iterations_per_second = max(60, Engine.get_frames_per_second())
		Engine.time_scale = max(1.0, Engine.iterations_per_second/10.0)
		
		if OS.get_name()=='X11_SHARED':
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
		
		$ActionLabel.text = "Action: "+str(agent_action)
		
		if env_action[0] == 1:
			$Anchor/PinJoint2D/RigidBody2D.reset = true
			time_elapsed = 0.0
			
		if env_action[1] == 1:
			get_tree().quit()
		
		$Anchor/PinJoint2D/RigidBody2D.torque = agent_action[0]*40.0
		
		$Timer.start(deltat)
		timeout = false

func _on_Timer_timeout():
	var observation = $Anchor/PinJoint2D/RigidBody2D.get_observation()
	var reward = [$Anchor/PinJoint2D/RigidBody2D.get_reward()]
	$ObservationLabel.text = "Observation: "+str(observation)
	$RewardLabel.text = "Reward: "+str(reward)
	$TimeLabel.text = "Time:"+str(time_elapsed)
	if OS.get_name()=='X11_SHARED':	
		mem.sendFloatArray("observation", observation)
		mem.sendFloatArray("reward", reward)
		mem.sendIntArray("done", [is_done()])
		sem_observation.post()
	
	time_elapsed += deltat
	timeout = true