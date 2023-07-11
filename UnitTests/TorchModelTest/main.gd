extends Node2D

@onready var model_data = load("res://test.tres")
var model:cTorchModel
# Called when the node enters the scene tree for the first time.
func _ready():
	var input:PackedFloat32Array
	for i in range(16):
		input.append(i)
	print(input)
	model = cTorchModel.new()
	model.set_data(model_data)
	var output = model.run(input)
	print(output)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
