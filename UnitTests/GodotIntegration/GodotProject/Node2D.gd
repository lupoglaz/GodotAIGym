extends Node2D

export var speed = 400
var screen_size
var mem = cSharedMemory.new()
var sem_action = cSharedMemorySemaphore.new()
var sem_observation = cSharedMemorySemaphore.new()
var action = [0, 1, 0, 0]
var observation = [1.0, 1.0]

func _ready():
	screen_size = get_viewport_rect().size
	sem_action.init("sem_action")
	sem_observation.init("sem_observation")
	set_physics_process(true)
	

func _physics_process(delta):
	
	sem_action.wait()
	action = mem.getIntArray("action")
	var cat = ""
	for i in action:
		cat += str(i)
	$Label.text = cat
	
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
	
	observation[0] = $Sprite.position.x
	observation[1] = $Sprite.position.y
	var cat_obs = "   "
	for i in observation:
		cat_obs += str(i)
	$Label.text += cat_obs
	mem.sendFloatArray("observation", observation)
	sem_observation.post()
