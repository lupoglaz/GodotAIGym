extends RigidBody2D

var reset = false
var init_transform
var init_velocity
var init_angular_velocity
var force = Vector2()
var init_joint_position

func _ready():
	init_transform = transform
	init_velocity = linear_velocity
	init_angular_velocity = angular_velocity
	set_physics_process(true)

func _integrate_forces(state):
	set_applied_force(force)
	
	if reset:
		state.transform = init_transform
		state.linear_velocity = init_velocity
		state.angular_velocity = init_angular_velocity
		reset=false
