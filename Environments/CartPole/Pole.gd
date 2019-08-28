extends RigidBody2D

var reset = false
var init_transform
var init_velocity
var init_angular_velocity
var init_tip_transform

func _ready():
	init_transform = transform
	init_velocity = linear_velocity
	init_angular_velocity = angular_velocity
	init_tip_transform = $StaticBody2D.transform

	#set_physics_process(true)

func _integrate_forces(state):
	if reset:
		$StaticBody2D.transform = init_tip_transform
		state.transform = init_transform
		state.linear_velocity = init_velocity
		state.angular_velocity = init_angular_velocity
		reset=false