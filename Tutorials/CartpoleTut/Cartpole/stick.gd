extends RigidBody2D

export var masscart = 1.0
export var masspole = 0.1
export var length = 0.5


var total_mass = (masspole + masscart)
var polemass_length = (masspole * length)


var whole
var env
var tau 

var reset = false
var init_origin
var init_rotation
var init_anchor
var init_transform

#State
var angular_vel
var angle = 0

# Called when the node enters the scene tree for the first time.
func _ready():
	whole = get_parent().get_parent()
	env = get_node("/root/Enviroment/")
	set_physics_process(true)

func _integrate_forces(state):
	#set_applied_torque(torque)
	print(init_anchor)
	print(init_origin)
	var T = Transform2D(0.0, init_anchor)
	var Tt = Transform2D(0.0, -init_anchor)
	var R = Transform2D(0.0, Vector2(0,0))
	state.set_transform(T*R*Tt*state.get_transform())
	
	
	
	if reset:
		
		
		var a = state.set_transform(init_transform)
		
		print(a)
		
		state.set_linear_velocity(Vector2(0.0, 0.0))
		state.set_angular_velocity(0.0)
		reset = false
	

	var gravity = state.get_total_gravity().y
	angular_vel = state.get_angular_velocity()
	angle = state.get_transform().get_rotation()
		
	var my_state = whole.get_state()
	var x = my_state[0]
	var x_dot = my_state[1]
	var theta = my_state[2]
	var theta_dot = my_state[3]
	var force = 0
	
	var costheta = cos(theta)
	var sintheta = sin(theta)
	
	
	if env.agent_action[0] > 0:
		force = whole.force_value
	if env.agent_action[0] < 0:
		force = - whole.force_value
	var temp = (force + polemass_length* pow(theta_dot,2) * sintheta) / total_mass
	var thetaacc = (gravity* sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * pow(costheta,2) / total_mass))
	var xacc = temp - polemass_length * thetaacc * costheta / total_mass
	state.set_angular_velocity(angular_vel + tau * thetaacc)
		
