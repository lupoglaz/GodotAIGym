extends RigidBody2D

var MAIN_ENGINE_MAG = 5
var SIDE_ENGINE_MAG = 1

func _ready():
	$LeftEngine.emitting = false
	$RightEngine.emitting = false
	$MainEngine.emitting = false
	pass
	
func _integrate_forces(state):
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