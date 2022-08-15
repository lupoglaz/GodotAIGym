extends Area2D

signal hit
signal coin

export var speed = 400 # How fast the player will move (pixels/sec).
var screen_size # Size of the game window.


func _ready():
	screen_size = get_viewport_rect().size
	hide()


func _process(delta):
	#print("_process")
	#print("get_parent().timeout: ", get_parent().timeout)
	
	if get_parent().timeout == true:
		#print("_process")
		get_parent().sem_physics.wait()
		var velocity = Vector2.ZERO # The player's movement vector.
		var speed_scale = 1

		if get_parent().mem.exists():
			if get_parent().action == 1:
				velocity.x += 1 
				
			if get_parent().action == 2:
				velocity.x -= 1
				
			if get_parent().action == 3:
				velocity.y += 1
				
			if get_parent().action == 4:
				velocity.y -= 1
				
			speed_scale = get_parent().speed_scale
		else:	
			if Input.is_action_pressed("move_right"):
				velocity.x += 1
			if Input.is_action_pressed("move_left"):
				velocity.x -= 1
			if Input.is_action_pressed("move_down"):
				velocity.y += 1
			if Input.is_action_pressed("move_up"):
				velocity.y -= 1
	
		if velocity.length() > 0:
			velocity = velocity.normalized() * speed
			$AnimatedSprite.play()
		else:
			$AnimatedSprite.stop()
	
		position += velocity * delta * speed_scale
		position.x = clamp(position.x, 0, screen_size.x)
		position.y = clamp(position.y, 0, screen_size.y)
	
		if velocity.x != 0:
			$AnimatedSprite.animation = "right"
			$AnimatedSprite.flip_v = false
			$AnimatedSprite.flip_h = velocity.x < 0
		elif velocity.y != 0:
			$AnimatedSprite.animation = "up"
			$AnimatedSprite.flip_v = velocity.y > 0
			
		get_parent().timeout = false
		get_parent().sem_physics.post()


func start(pos):
	position = pos
	show()
	$CollisionShape2D.disabled = false


func _on_Player_body_entered(_body):
	#print("_on_Player_body_entered")
	#print("_body.name: ", _body.name)
	
	#hide() # Player disappears after being hit.
	if "Mob" in _body.name:
		hide()
		emit_signal("hit")
		
		## Must be deferred as we can't change physics properties on a physics callback.
		$CollisionShape2D.set_deferred("disabled", true)
	elif "Coin" in _body.name:
		emit_signal("coin")
		_body.queue_free()
	
