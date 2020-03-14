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

var reached_goal = false

#lander state
var pos_x
var pos_y
var vel_x
var vel_y
var angular_vel
var angle

var reset = false
var init_transform


func _ready():
	
	#Getting shapes local indexes
	var shape_owners = get_shape_owners()
	$RightLegShape.get_shape().set_meta("name", "right_leg")
	$LeftLegShape.get_shape().set_meta("name", "left_leg")
	$BodyShape.get_shape().set_meta("name", "body")
	for owner_id in shape_owners:
		for shape_id in shape_owner_get_shape_count(owner_id):
			var shape = shape_owner_get_shape(owner_id, shape_id)
			if shape.get_meta("name") == "right_leg":
				right_leg_idx = shape_owner_get_shape_index(owner_id, shape_id)
			if shape.get_meta("name") == "left_leg":
				left_leg_idx = shape_owner_get_shape_index(owner_id, shape_id)
			if shape.get_meta("name") == "body":
				body_idx = shape_owner_get_shape_index(owner_id, shape_id)
				
	init_transform = transform
		
func _integrate_forces(state):
	if reset:
		var new_transform = init_transform
		new_transform = new_transform.rotated(rand_range(-0.2, 0.2))
		new_transform = new_transform.translated(Vector2(rand_range(-300.0, 300.0), 0.0))
		state.set_transform(new_transform)
		state.set_linear_velocity(Vector2(0.0, 0.0))
		state.set_angular_velocity(0.0)
		reset = false
	
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
	
	
	if left_leg_collided and right_leg_collided and get_parent().in_landing_area:
		reached_goal = true
	else:
		reached_goal = false
		
