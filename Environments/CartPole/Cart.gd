extends RigidBody2D

var reset = false
var init_transform
var init_velocity
var init_angular_velocity
var force = Vector2()
var init_joint_position
var add_shift = 0

func _ready():
	init_transform = transform
	init_velocity = linear_velocity
	init_angular_velocity = angular_velocity
	#set_physics_process(true)

func _integrate_forces(state):
	if reset:
		state.set_transform(init_transform)
		state.linear_velocity = init_velocity
		state.angular_velocity = init_angular_velocity
		state.transform.origin.x += add_shift
		force = Vector2(0.0, 0.0)
		reset=false
	
	set_applied_force(force)
