extends RigidBody2D

var reset = false
var init_transform
var init_velocity
var init_angular_velocity
var init_tip_transform
var add_shift = 0
var add_angle = 0

func _ready():
	init_transform = transform
	init_velocity = linear_velocity
	init_angular_velocity = angular_velocity
	#set_physics_process(true)

func _integrate_forces(state):
	if reset:
		#$StaticBody2D.transform = init_tip_transform
		state.set_transform(init_transform)
		state.linear_velocity = init_velocity
		state.angular_velocity = init_angular_velocity + add_angle
		state.transform.origin.x += add_shift
		reset=false