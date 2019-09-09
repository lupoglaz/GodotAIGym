extends Node2D

var FixedDelta = 0.1

var observation = [0.0, 0.0, 0.0, 0.0]
var agent_action = [0]
var env_action = [0, 0]

var force_mag = 500.0

var sem_action
var sem_observation
var sem_reset
var mem

var init_joint_position
var init_grove_position
var count_down = -1

func _ready():
	if Global.release:
		sem_action = cSharedMemorySemaphore.new()
		sem_observation = cSharedMemorySemaphore.new()
		sem_reset = cSharedMemorySemaphore.new()
		mem = cSharedMemory.new()
		sem_action.init("sem_action")
		sem_observation.init("sem_observation")

	
	init_joint_position = $Joint.position
	init_grove_position = $GrooveJoint2D.get_initial_offset()
	
	
	print("Initialized environment")
	#set_physics_process(true)
	Engine.time_scale = 5

func is_done():
	if abs($Cart.transform.origin.x - $Cart.transform.origin.x)>500.0:
		return 1
	if abs($Pole.transform.get_rotation()) > 10.0*PI/180.0:
		return 1
	return 0

func get_count_down():
	return int(floor(Engine.get_frames_per_second()/(30.0*Engine.time_scale)))
	
#func _physics_process(delta):
func _process(delta):
	#Engine.time_scale = Engine.get_frames_per_second()/60.0
	if $Cart.reset or $Pole.reset:
		return
	
	if Global.release:
		if count_down == -1:
			sem_action.wait()
			agent_action = mem.getIntArray("agent_action")
			env_action = mem.getIntArray("env_action")
			count_down = get_count_down()
	else:
		if Input.is_action_pressed("ui_right"):
			agent_action[0] = 0
		if Input.is_action_pressed("ui_left"):
			agent_action[0] = 1
		if Input.is_action_pressed("ui_down"):
			env_action[0] = 1
			env_action[1] = 0
		if Input.is_action_pressed("ui_up"):
			env_action[0] = 0
			env_action[1] = 1
		
		
	if env_action[0] == 1:
		$Cart.reset = true
		$Pole.reset = true
		$Pole.add_shift = rand_range(-30.0, 30.0)
		$Cart.add_shift = $Pole.add_shift
		$Pole.add_angle = rand_range(-0.005, 0.005)
		env_action[0] = 0
		agent_action[0] = 0
		count_down = get_count_down()
		
	#act
	if agent_action[0] == 0:
		$Cart.force = Vector2(force_mag, 0)
		$Cart/CPUParticles2D.emitting = true
		$Cart/CPUParticles2D2.emitting = false
	elif agent_action[0] == 1:
		$Cart.force = Vector2(-force_mag, 0)
		$Cart/CPUParticles2D2.emitting = true
		$Cart/CPUParticles2D.emitting = false
	else:
		print("unknown action")
		
	observation[0] = 5.0*($Cart.transform.origin.x - $Cart.init_transform.origin.x)/500.0
	observation[1] = $Cart.linear_velocity.x
	observation[2] = $Pole.transform.get_rotation()
	observation[3] = 5.0*($Pole.angular_velocity * 250.0 + $Cart.linear_velocity.x)/500.0
	
	if Global.release:
		if count_down == 0:
			mem.sendFloatArray("observation", observation)
			mem.sendFloatArray("reward", [get_count_down()])
			mem.sendIntArray("done", [is_done()])
			sem_observation.post()
			count_down -= 1
		else:
			count_down -= 1
	
	
