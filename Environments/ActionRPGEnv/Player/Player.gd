extends KinematicBody2D

export var ACCELERATION = 500
export var MAX_SPEED = 80
export var ROLL_SPEED = 120
export var FRICTION = 500
var velocity = Vector2.ZERO
var roll_vec = Vector2.DOWN

onready var animationPlayer = $AnimationPlayer
onready var animationTree = $AnimationTree
onready var animationState = animationTree.get("parameters/playback")
onready var swordHitbox = $HitboxPivot/SwordHitbox
onready var hurtbox = $Hurtbox
onready var blinkAnimationPlayer = $BlinkAnimationPlayer

var stats = PlayerStats
const PlayerHurtSound = preload("res://Player/PlayerHurtSound.tscn")


enum {
	MOVE,
	ROLL,
	ATACK
}
var state = MOVE


func _ready():
	randomize()
	stats.connect("no_health", self, "queue_free")
	animationTree.active = true

func _process(delta):
	match state:
		MOVE:
			move_state(delta)
		ROLL:
			roll_state(delta)
		ATACK:
			atack_state(delta)

func move_state(delta):
	var input_vec = Vector2.ZERO
	input_vec.x = Input.get_action_strength("ui_right") - Input.get_action_strength("ui_left")
	input_vec.y = Input.get_action_strength("ui_down") - Input.get_action_strength("ui_up")
	input_vec = input_vec.normalized()
	if input_vec != Vector2.ZERO:
		roll_vec = input_vec
		swordHitbox.knockback_vector = input_vec
		animationTree.set("parameters/Idle/blend_position", input_vec)
		animationTree.set("parameters/Run/blend_position", input_vec)
		animationTree.set("parameters/Atack/blend_position", input_vec)
		animationTree.set("parameters/Roll/blend_position", input_vec)
		animationState.travel("Run")
		velocity = velocity.move_toward(input_vec*MAX_SPEED, ACCELERATION*delta)
	else:
		animationState.travel("Idle")
		velocity = velocity.move_toward(Vector2.ZERO, FRICTION*delta)
	
	move()
	
	if Input.is_action_just_pressed("atack"):
		state = ATACK
		
	if Input.is_action_just_pressed("roll"):
		state = ROLL
		
func roll_state(delta):
	velocity = roll_vec * ROLL_SPEED
	animationState.travel("Roll")
	move()

func atack_state(delta):
	velocity = Vector2.ZERO
	animationState.travel("Atack")

func move():
	velocity = move_and_slide(velocity)

func atack_animation_finished():
	state = MOVE
	
func roll_animation_finished():
	velocity = Vector2.ZERO
	state = MOVE
	


func _on_Hurtbox_area_entered(area):
	stats.health -= area.damage
	hurtbox.start_invincibility(0.6)
	hurtbox.create_hit_effect()
	var playerHurtSound = PlayerHurtSound.instance()
	get_tree().current_scene.add_child(playerHurtSound)


func _on_Hurtbox_invincibility_started():
	blinkAnimationPlayer.play("Start")


func _on_Hurtbox_invincibility_ended():
	blinkAnimationPlayer.play("Stop")
