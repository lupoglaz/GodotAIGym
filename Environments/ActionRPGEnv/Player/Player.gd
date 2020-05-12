extends KinematicBody2D

const ACCELERATION = 500
const MAX_SPEED = 100
const FRICTION = 500
var velocity = Vector2.ZERO

onready var animationPlayer = $AnimationPlayer
onready var animationTree = $AnimationTree
onready var animationState = animationTree.get("parameters/playback")

enum {
	MOVE,
	ROLL,
	ATACK
}
var state = MOVE


func _ready():
	animationTree.active = true

func _process(delta):
	match state:
		MOVE:
			move_state(delta)
		ROLL:
			pass
		ATACK:
			atack_state(delta)

func move_state(delta):
	var input_vec = Vector2.ZERO
	input_vec.x = Input.get_action_strength("ui_right") - Input.get_action_strength("ui_left")
	input_vec.y = Input.get_action_strength("ui_down") - Input.get_action_strength("ui_up")
	input_vec = input_vec.normalized()
	if input_vec != Vector2.ZERO:
		animationTree.set("parameters/Idle/blend_position", input_vec)
		animationTree.set("parameters/Run/blend_position", input_vec)
		animationTree.set("parameters/Atack/blend_position", input_vec)
		animationState.travel("Run")
		velocity = velocity.move_toward(input_vec*MAX_SPEED, ACCELERATION*delta)
	else:
		animationState.travel("Idle")
		velocity = velocity.move_toward(Vector2.ZERO, FRICTION*delta)
	
	velocity = move_and_slide(velocity)
	
	if Input.is_action_just_pressed("atack"):
		state = ATACK
	
func atack_state(delta):
	velocity = Vector2.ZERO
	animationState.travel("Atack")

func atack_animation_finished():
	state = MOVE
