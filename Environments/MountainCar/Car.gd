extends RigidBody2D

export var force = 3

var ground
var env
var pos
var vel
var rot
# Called when the node enters the scene tree for the first time.
func _ready():
	ground = get_node('/root/Enviroment/Ground')
	env = get_parent()
	set_physics_process(true)

func get_state():
	#var my_pos = (-pos + 365)/163 *1.8 - 1.2
	return [pos,vel, rot]



func _integrate_forces(state):
	var glob = state.get_transform()
	print()
	if env.env_action[0] == 1:
		var now = ground.maxi_car - Vector2(rand_range(-50.0, 50.0), 20)
		state.set_transform(Transform2D(0, now))
		state.set_linear_velocity(Vector2(rand_range(-50,50), 0.0))
		state.set_angular_velocity(0.0)
	
	pos = glob.origin.y
	vel = state.get_linear_velocity().x
	rot = get_rotation()/(PI)
	
	
	var agent_action = get_parent().agent_action
	if agent_action[0] > 0:
		apply_impulse(Vector2(), glob.basis_xform(Vector2(force,0)))
	if agent_action[0] < 0:
		apply_impulse(Vector2(), glob.basis_xform(Vector2(-force,0)))
