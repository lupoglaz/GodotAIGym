extends Node2D

export var speed = 400
var screen_size
#var mem = cSharedMemory.new()
var sem = cSharedMemorySemaphore.new()
var action = [1, 0, 0, 0]

func _ready():
	screen_size = get_viewport_rect().size
	#action = mem.getArray("action")
	for i in action:
		$Label.text += str(i)
	#print(action)
	sem.init("sem3")

func _process(delta):
	
	sem.wait()
		
	var velocity = Vector2()
	if action[0] == 1:
		velocity.x += 1
	if action[1] == 1:
		velocity.x -= 1
	if action[2] == 1:
		velocity.y += 1
	if action[3] == 1:
		velocity.y -= 1
	if velocity.length() > 0:
		velocity = velocity.normalized() * speed

	$Sprite.position += velocity * delta
	$Sprite.position.x = clamp($Sprite.position.x, 0, screen_size.x)
	$Sprite.position.y = clamp($Sprite.position.y, 0, screen_size.y)
	#mem.sendArray("observation", [position.x, position.y])
	#sem.post()
