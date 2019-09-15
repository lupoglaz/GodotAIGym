extends RigidBody2D

var torque = 0
var reset = false
var init_origin
var init_rotation
var init_angular_velocity
var init_linear_velicity

var rng = RandomNumberGenerator.new()

func _ready():
	set_physics_process(true)

func _integrate_forces(state):
	set_applied_torque(torque)
	if reset:
		state.set_transform(Transform2D(init_rotation + rng.randf_range(-0.1, 0.1), init_origin))
		state.set_angular_velocity(init_angular_velocity + rng.randf_range(-10, 10))
		state.set_linear_velocity(init_linear_velicity)
		reset = false

func get_observation():
	var observation = [0.0, 0.0, 0.0, 0.0]
	observation[0] = angular_velocity
	observation[1] = transform.get_rotation()
	observation[2] = torque
	observation[3] = transform.basis_xform($CollisionShape2D/Position2D.transform.get_origin()).y
	return observation