extends Node2D


# Declare member variables here. Examples:
# var a = 2
# var b = "text"

var target_delta = 0.05
var average_delta = 0.0
var frames = 0
var time_elapsed = 0

var obs_delta = 1.0
var mem
var sem_observation
var sem_action
var env_action

var timeout = true
var sem_delta = 0.0

var current_time = 0.0
var kin_body = []
var time = []
var its = []
var fps = []
var prev_time = 0.0

func _ready():
	Engine.set_target_fps(0)
	mem = cSharedMemory.new()
	if mem.exists():
		sem_action = cSharedMemorySemaphore.new()
		sem_observation = cSharedMemorySemaphore.new()
		
		sem_action.init("sem_action")
		sem_observation.init("sem_observation")
		
		print("Running as OpenAIGym environment")
		
	
func _physics_process(delta):
	$KinematicBody2D.move_and_slide(Vector2(2.0,0))
	average_delta += delta
	frames += 1
	
	time.append(current_time)
	kin_body.append($KinematicBody2D.position.x)
	its.append(Engine.get_iterations_per_second())
	fps.append(Engine.get_frames_per_second())
	current_time += delta
	
	if mem.exists() and timeout:
		
		var time_start = OS.get_ticks_usec()
		sem_action.wait()
		var time_end = OS.get_ticks_usec()
		sem_delta = time_end - time_start
		timeout = false
		$Timer.start(obs_delta)
		save()
	
func _process(delta):
	var cur_time = OS.get_ticks_usec()
	var fps_est = 1000000.0/(cur_time - prev_time - sem_delta)
	Engine.set_iterations_per_second(fps_est)
	Engine.set_time_scale(Engine.get_iterations_per_second()*target_delta)
	if sem_delta>0:
		print(fps_est, " | ",Engine.get_iterations_per_second(), " | ", sem_delta)
	sem_delta = 0.0
	prev_time = cur_time


func _on_Timer_timeout():
	timeout = true
	if mem.exists():# and timeout:
		var obs = [$KinematicBody2D.position.x, Engine.get_iterations_per_second(), Engine.get_frames_per_second(), 
		Engine.get_time_scale(), average_delta/frames, time_elapsed]
		mem.sendFloatArray("observation", obs)
		sem_observation.post()
		average_delta = 0.0
		frames = 0
		time_elapsed += 1.0
		

func save():
	var log_file = File.new()
	log_file.open("res://log.json", File.WRITE)
	var dict = {
		"Kin" : kin_body,
		"Its" : its,
		"Fps" : fps,
		"t" : time
	}
	log_file.store_line(to_json(dict))
