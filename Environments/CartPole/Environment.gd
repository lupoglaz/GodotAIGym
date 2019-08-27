extends Node2D


var FixedDelta = 0.1
var action = [0, 0]
var observation = []
var force_mag = 200.0

var init_joint_position

func _ready():
	init_joint_position = $Joint.position
	set_physics_process(true)

func _physics_process(delta):
	if Input.is_action_pressed('ui_right'):
		$Cart.force = Vector2(force_mag, 0)
		
	elif Input.is_action_pressed('ui_left'):
		$Cart.force = Vector2(-force_mag, 0)
	elif Input.is_action_pressed("ui_down"):
		$Cart.reset = true
		$Pole.rest = true
		$Joint.position = init_joint_position
	else:
		$Cart.force = Vector2(0.0, 0.0)
	
	
