extends RigidBody2D

var MAIN_ENGINE_MAG = 5
var SIDE_ENGINE_MAG = 1

var left_leg_idx
var right_leg_idx
var body_idx

var left_leg_collided = false
var right_leg_collided = false
var body_collided = false

var left_engine_power = 1.0
var right_engine_power = 1.0
var main_engine_power = 1.0

var rest = false

#lander state
var pos_x
var pos_y
var vel_x
var vel_y
var angular_vel
var angle

func _ready():
	
	#Getting shapes local indexes
	var shape_owners = get_shape_owners()
	$RightLegShape.get_shape().set_meta("name", "right_leg")
	$LeftLegShape.get_shape().set_meta("name", "left_leg")
	$BodyShape.get_shape().set_meta("name", "body")
	for owner_id in shape_owners:
		print("Owner:",owner_id, "Num shapes:", shape_owner_get_shape_count(owner_id))
		for shape_id in shape_owner_get_shape_count(owner_id):
			var shape = shape_owner_get_shape(owner_id, shape_id)
			if shape.get_meta("name") == "right_leg":
				right_leg_idx = shape_owner_get_shape_index(owner_id, shape_id)
			if shape.get_meta("name") == "left_leg":
				left_leg_idx = shape_owner_get_shape_index(owner_id, shape_id)
			if shape.get_meta("name") == "body":
				body_idx = shape_owner_get_shape_index(owner_id, shape_id)
	
func _integrate_forces(state):
	
	var glob = state.get_transform()
	if $MainEngine.emitting:
		var force = glob.basis_xform(Vector2(0.0, -MAIN_ENGINE_MAG))
		var offset = glob.basis_xform(Vector2(0, 25))
		apply_impulse(offset, main_engine_power*force)
		
	if $LeftEngine.emitting:
		var force = glob.basis_xform(Vector2(SIDE_ENGINE_MAG, 0.0))
		var offset = glob.basis_xform(Vector2(-25, -25))
		apply_impulse(offset, left_engine_power*force)
		
	if $RightEngine.emitting:
		var force = glob.basis_xform(Vector2(-SIDE_ENGINE_MAG, 0.0))
		var offset = glob.basis_xform(Vector2(25, -25))
		apply_impulse(offset, right_engine_power*force)
		
	#Getting colliding shapes indexes
	right_leg_collided = false
	left_leg_collided = false
	body_collided = false
	for i in range(state.get_contact_count()):
		var shape_idx = state.get_contact_local_shape(i)
		if right_leg_idx == shape_idx:
			right_leg_collided = true
		if left_leg_idx == shape_idx:
			left_leg_collided = true
		if body_idx == shape_idx:
			body_collided = true
	
	pos_x = glob.origin.x
	pos_y = glob.origin.y
	vel_x = state.get_linear_velocity().x
	vel_y = state.get_linear_velocity().y
	angular_vel = state.get_angular_velocity()
	angle = glob.get_rotation()
	
	if state.get_linear_velocity().length()<1.0 and state.get_angular_velocity()<0.01:
		rest = true
	else:
		rest = false