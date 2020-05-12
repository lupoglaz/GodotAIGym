extends KinematicBody2D

var knockback = Vector2.ZERO
onready var stats = $Stats

const EnemyDeathEffect = preload("res://Effects/EnemyDeathEffect.tscn")

func _process(delta):
	knockback = knockback.move_toward(Vector2.ZERO, 200*delta)
	knockback = move_and_slide(knockback)

func _on_Hurtbox_area_entered(area):
	knockback = area.knockback_vector * 150
	stats.health -= area.damage
	
func _on_Stats_no_health():
	queue_free()
	var enemyDeathEffect = EnemyDeathEffect.instance()
	get_parent().add_child(enemyDeathEffect)
	enemyDeathEffect.global_position = global_position
