extends RigidBody2D

var torque = 0
var reset = false
var init_origin
var init_rotation
var init_angular_velocity
var init_anchor

var rng = RandomNumberGenerator.new()

func _ready():
	set_physics_process(true)

func _integrate_forces(state):
	set_applied_torque(torque)
	if reset:
		var T = Transform2D(0.0, init_anchor)
		var Tt = Transform2D(0.0, -init_anchor)
		var R = Transform2D(rng.randf_range(-PI, PI), Vector2(0,0))
#		var R = Transform2D(0.0, Vector2(0,0))
		state.set_transform(T*R*Tt*state.transform)
		state.set_angular_velocity(init_angular_velocity)
		state.set_linear_velocity(Vector2(0,0))
		reset = false

func get_observation():
	var observation = [0.0, 0.0, 0.0]
	observation[0] = -cos(transform.get_rotation())
	observation[1] = sin(transform.get_rotation())
	observation[2] = angular_velocity
	#observation[3] = transform.basis_xform($CollisionShape2D/Position2D.transform.get_origin()).y
	return observation
	
func get_reward():
	var th = fmod((transform.get_rotation() + 2*PI), (2*PI)) - PI
	return -(th*th + .1*angular_velocity*angular_velocity + 0.001*(torque/200.0)*(torque/200.0))
