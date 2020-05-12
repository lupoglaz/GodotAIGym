extends Node2D


# Declare member variables here. Examples:
# var a = 2
# var b = "text"


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	if Input.is_action_just_pressed("atack"):
		var grassEffectScene = load("res://Effects/GrassEffect.tscn")
		var grassEffect = grassEffectScene.instance()
		var parent = get_parent()
		parent.add_child(grassEffect)
		grassEffect.position = self.position
		queue_free()
