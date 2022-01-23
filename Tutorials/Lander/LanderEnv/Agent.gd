extends RigidBody2D


# Declare member variables here. Examples:
# var a = 2
# var b = "text"

var main_engine_power = 1.0
var left_engine_power = 1.0
var right_engine_power = 1.0
var MAIN_ENGINE_MAG = 5
var SIDE_ENGINE_MAG = 1

var right_leg_idx 
var left_leg_idx
var body_idx

var right_leg_collided
var left_leg_collided
var body_collided

var num_rays = 10
var vis_length = 500.0
var angle_inc = PI/float(num_rays)
var vision = []
var vision_lines = []


func act(action):
	$MainEngine.emitting = false
	$LeftEngine.emitting = false
	$RightEngine.emitting = false
	if action[0] > 0.5:
		$MainEngine.emitting = true
		main_engine_power = clamp(abs(action[0]), 0.0, 1.0)
	if action[1] > 0.5:
		$LeftEngine.emitting = true
		left_engine_power = clamp(abs(action[1]), 0.0, 1.0)
	if action[1] < -0.5:
		$RightEngine.emitting = true
		right_engine_power = clamp(abs(action[1]), 0.0, 1.0)

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
			
	var space_state = get_world_2d().direct_space_state
	var origin = Vector2(0,0)
	var origin_global = glob.xform(origin)
	for i in range(num_rays):
		var angle = i*angle_inc
		var target = Vector2(cos(angle), sin(angle))*vis_length
		var target_global = glob.xform(target)
		var result = space_state.intersect_ray(origin_global, target_global, [self], collision_mask)
		vision[i] = -1.0
		vision_lines[i].set_point_position(0, origin)
		vision_lines[i].set_point_position(1, target)
		if len(result):
			vision_lines[i].set_point_position(1, glob.xform_inv(result["position"]))
			if result["collider"].name == 'Ground':
				vision[i] = 1.0 - (result["position"] - origin_global).length()/vis_length

# Called when the node enters the scene tree for the first time.
func _ready():
	$RightLeg.get_shape().set_meta("name", "right_leg")
	$LeftLeg.get_shape().set_meta("name", "left_leg")
	$Body.get_shape().set_meta("name", "body")
	for owner_id in get_shape_owners():
		for shape_id in shape_owner_get_shape_count(owner_id):
			var shape = shape_owner_get_shape(owner_id, shape_id)
			if shape.get_meta("name") == "right_leg":
				right_leg_idx = shape_owner_get_shape_index(owner_id, shape_id)
			if shape.get_meta("name") == "left_leg":
				left_leg_idx = shape_owner_get_shape_index(owner_id, shape_id)
			if shape.get_meta("name") == "body":
				body_idx = shape_owner_get_shape_index(owner_id, shape_id)
				
	vision = []
	vision_lines = []
	for i in range(num_rays):
		vision.append(-1)
		var line = Line2D.new()
		line.add_point(Vector2(0,0))
		line.add_point(Vector2(0,0))
		line.width = 3
		add_child(line)
		vision_lines.append(line)
				
func get_observation(goal_vec2):
	var SCALE_X = get_viewport().size.x * 0.5
	var SCALE_Y = get_viewport().size.y * 0.5
	var SCALE = (SCALE_X + SCALE_Y)/2.0
	var lin_vel = get_linear_velocity()
	var dir = goal_vec2 - get_global_transform().get_origin()
	
	var observation = [
		dir.x/(dir.length() + 0.001),
		dir.y/(dir.length() + 0.001),
		dir.length()/SCALE,
		lin_vel.x/(lin_vel.length() + 0.001),
		lin_vel.y/(lin_vel.length() + 0.001),
		lin_vel.length()/(SCALE/10.0),
		cos(get_rotation()),
		sin(get_rotation()),
		get_angular_velocity()/(PI/10.0),
		1 if left_leg_collided else 0,
		1 if right_leg_collided else 0
	] + vision
	return observation
