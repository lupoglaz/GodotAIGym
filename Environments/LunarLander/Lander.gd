extends RigidBody2D

var MAIN_ENGINE_MAG = 5
var SIDE_ENGINE_MAG = 1

func _ready():
	$LeftEngine.emitting = false
	$RightEngine.emitting = false
	$MainEngine.emitting = false
	print(get_shape_owners())
	pass
	
func _integrate_forces(state):
	#var col = get_colliding_bodies()
	
	var num_contacts = state.get_contact_count()
	for i in range(num_contacts):
		var shape_id = state.get_contact_local_shape(i)
		var owner_id = state.shape_find_owner(shape_id)
		
		print()
		if $RightLegShape.get_shape() == shape_owner_get_shape(owner_id, shape_id):
			print("right leg colliding")
		if $LeftLegShape.get_shape() == shape_owner_get_shape(owner_id, shape_id):
			print("left leg colliding")
		
	
		
	var glob = state.get_transform()
	if $MainEngine.emitting:
		var force = glob.basis_xform(Vector2(0.0, -MAIN_ENGINE_MAG))
		var offset = glob.basis_xform(Vector2(0, 25))
		apply_impulse(offset, force)
		
	if $LeftEngine.emitting:
		var force = glob.basis_xform(Vector2(SIDE_ENGINE_MAG, 0.0))
		var offset = glob.basis_xform(Vector2(-25, -25))
		apply_impulse(offset, force)
		
	if $RightEngine.emitting:
		var force = glob.basis_xform(Vector2(-SIDE_ENGINE_MAG, 0.0))
		var offset = glob.basis_xform(Vector2(25, -25))
		apply_impulse(offset, force)