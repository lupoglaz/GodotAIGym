extends Node2D


# Declare member variables here. Examples:
# var a = 2
# var b = "text"
var target_delta = 0.05
var average_delta = 0.0
var frames = 0

var max_time = 10
var time_elapsed = 0

var kin_body = []
var rig_body = []
var time = []

# Called when the node enters the scene tree for the first time.
func _ready():
	Engine.set_target_fps(0)

func _physics_process(delta):
	$KinematicBody2D.move_and_slide(Vector2(2.0,0))
	average_delta += delta
	frames += 1

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	Engine.set_iterations_per_second(Engine.get_frames_per_second())
	Engine.set_time_scale(Engine.get_frames_per_second()*target_delta)


func _on_Timer_timeout():
	var info = str($KinematicBody2D.position.x) + " " + str(Engine.get_iterations_per_second())
	info += " " + str(Engine.get_frames_per_second())
	info += " " + str(average_delta/frames)
	info += " " + str(time_elapsed)
	print(info)
	average_delta = 0.0
	frames = 0
	time_elapsed += 1.0
	kin_body.append($KinematicBody2D.position.x)
	rig_body.append($RigidBody2D.position.y)
	time.append(time_elapsed)
	
	if time_elapsed > max_time:
		done()
	
func done():
	var log_file = File.new()
	log_file.open("res://log.json", File.WRITE)
	var dict = {
		"Rig" : rig_body,
		"Kin" : kin_body,
		"t" : time
	}
	log_file.store_line(to_json(dict))
	get_tree().quit()

