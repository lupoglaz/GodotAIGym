extends KinematicBody2D

var knockback = Vector2.ZERO
onready var stats = $Stats
onready var sprite = $AnimatedSprite
const EnemyDeathEffect = preload("res://Effects/EnemyDeathEffect.tscn")

export var ACCELERATION = 300
export var MAX_SPEED = 50
export var FRICTION = 200

onready var playerDetectionZone = $PlayerDetectionZone
onready var hurtbox = $Hurtbox
onready var softCollision = $SoftCollision
onready var wanderController = $WanderController
onready var animationPlayer = $AnimationPlayer

enum {
	IDLE,
	WANDER,
	CHASE
}
var velocity = Vector2.ZERO
var state = IDLE

func _ready():
	state = pick_rand_state([IDLE, WANDER])

func _process(delta):
	knockback = knockback.move_toward(Vector2.ZERO, 200*delta)
	knockback = move_and_slide(knockback)
	
	match state:
		IDLE:
			velocity = velocity.move_toward(Vector2.ZERO, FRICTION*delta)
			seek_player()
			if wanderController.get_time_left() == 0:
				state = pick_rand_state([IDLE, WANDER])
				wanderController.start_wander_timer(rand_range(1,3))
		WANDER:
			seek_player()
			if wanderController.get_time_left() == 0:
				state = pick_rand_state([IDLE, WANDER])
				wanderController.start_wander_timer(rand_range(1,3))
			var direction = global_position.direction_to(wanderController.target_position)
			if global_position.distance_to(wanderController.target_position) >= MAX_SPEED*delta:
				accelerate_toward(wanderController.target_position, delta)
			else:
				state = pick_rand_state([IDLE, WANDER])
				wanderController.start_wander_timer(rand_range(1,3))
			
		CHASE:
			var player = playerDetectionZone.player
			if player != null:
				accelerate_toward(player.global_position, delta)
			else:
				state = IDLE
			
	if softCollision.is_colliding():
		velocity += softCollision.get_push_vector() * delta * 400
	velocity = move_and_slide(velocity)

func accelerate_toward(target, delta):
	var direction = global_position.direction_to(target)
	velocity = velocity.move_toward(direction*MAX_SPEED, ACCELERATION*delta)
	sprite.flip_h = velocity.x < 0

func seek_player():
	if playerDetectionZone.can_see_player():
		state = CHASE

func pick_rand_state(state_list):
	state_list.shuffle()
	return state_list.pop_front()
	
func _on_Hurtbox_area_entered(area):
	knockback = area.knockback_vector * 150
	stats.health -= area.damage
	hurtbox.create_hit_effect()
	hurtbox.start_invincibility(0.3)
	
func _on_Stats_no_health():
	queue_free()
	var enemyDeathEffect = EnemyDeathEffect.instance()
	get_parent().add_child(enemyDeathEffect)
	enemyDeathEffect.global_position = global_position


func _on_Hurtbox_invincibility_started():
	animationPlayer.play("Start")

func _on_Hurtbox_invincibility_ended():
	animationPlayer.play("Stop")
