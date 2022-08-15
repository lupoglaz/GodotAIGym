extends RigidBody2D

onready var animationPlayer = $AnimationPlayer

# Called when the node enters the scene tree for the first time.
func _ready():
	#$Sprite.playing = true
	var coin_types = 0
	animationPlayer.play("Side1")
	

func _on_VisibilityNotifier2D_screen_exited():
	queue_free()


func _on_Timer_timeout():
	print("Coin _on_Timer_timeout()")
	queue_free()
