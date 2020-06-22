extends RigidBody2D



#State
var pos_x
var vel_x
var env
export var force_value = 5
export var scale_speeds = 0.1
var init_transform
var reset = false



# Called when the node enters the scene tree for the first time.
func _ready():
	env = get_parent().get_parent()
	
	
	
	set_physics_process(true)
	



func _integrate_forces(state):
	
	if reset:
		var new_transform= init_transform
		
		state.set_transform(new_transform)
		state.set_linear_velocity(Vector2(0.0, 0.0))
		
		
		$PinJoint2D/stick.reset = true
		reset = false
	
	vel_x = state.get_linear_velocity().x
	pos_x = get_position().x
	
	var glob = state.get_transform()
	
	
	var offset = glob.basis_xform(Vector2(0, 0))
	if env.agent_action[0] > 0:
		var force = glob.basis_xform(Vector2(force_value, 0))
		apply_impulse(offset, force)
	if env.agent_action[0] < 0:
		var force = glob.basis_xform(Vector2((-force_value), 0))
		apply_impulse(offset, force)
	
	
func get_state():
	if pos_x != null:
		var scaled_pos = pos_x/(get_viewport().size.x/2)*2.4
		var state = [scaled_pos,vel_x*scale_speeds, 
		$PinJoint2D/stick.angle, $PinJoint2D/stick.angular_vel*scale_speeds] 
		return state
	else:
		return null
