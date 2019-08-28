extends Node2D


var FixedDelta = 0.1

var observation = [0.0, 0.0, 0.0, 0.0]
var agent_action = [0]
var env_action = [0, 0]

var force_mag = 200.0
var sem_action = cSharedMemorySemaphore.new()
var sem_observation = cSharedMemorySemaphore.new()
var mem = cSharedMemory.new()

var init_joint_position

func _ready():
	init_joint_position = $Joint.position
	sem_action.init("sem_action")
	sem_observation.init("sem_observation")
	print("Initialized environment")
	set_physics_process(true)
	#Engine.time_scale = 10

func _physics_process(delta):
#func _process(delta):
	
	sem_action.wait()
	agent_action = mem.getIntArray("agent_action")
	env_action = mem.getIntArray("env_action")
	#reset
	if env_action[0] == 1:
		$Cart.reset = true
		$Pole.rest = true
		$Joint.position = init_joint_position
		return
	#exit
	elif env_action[1] == 1:
		get_tree().quit()
	
	#act
	Engine.iterations_per_second = Engine.get_frames_per_second()
	$Label.text = "FPS: " + str(Engine.get_frames_per_second())
	if agent_action[0] == 0:
		$Cart.force = Vector2(force_mag, 0)	
	elif agent_action[0] == 1:
		$Cart.force = Vector2(-force_mag, 0)
	else:
		print("unknown action")
		
	observation[0] = ($Cart.transform.x - $Cart.init_transform.x)/100.0
	observation[1] = $Cart.linear_velocity/100.0
	observation[2] = $Pole.transform.get_rotation()
	observation[3] = $Pole/StaticBody2D.constant_linear_velocity/100.0
	
	mem.sendFloatArray("observation", observation)
	mem.sendFloatArray("reward", [1.0])
	mem.sendIntArray("done", [0])
	sem_observation.post()
	
	
