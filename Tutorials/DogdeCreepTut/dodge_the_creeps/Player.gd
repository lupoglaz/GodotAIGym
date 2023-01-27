extends Area2D

signal hit
signal coin

export var speed = 600 # How fast the player will move (pixels/sec).
var screen_size # Size of the game window.
var velocity = Vector2.ZERO # The player's movement vector.

var deltat = 0.05
var timeout = true

var agent_action = [0.0]
var env_action = [0,0]
var action = 0

func _ready():
	screen_size = get_viewport_rect().size
	hide()


func new_game_player(score):
	$HUD.update_score(score)
	$HUD.show_message("Get Ready")
	$HUD.hide_game_over()
	

func update_score_player(score):	
	#get_parent().score += 1
	$HUD.update_score(score)


func _process(delta):
	print("delta: ", delta)
	#print("_process")
	if timeout == true:
		get_parent().sem_physics.wait()
		var speed_scale = 1

		if get_parent().mem.exists():
			get_parent().sem_action.wait()
			agent_action = get_parent().agent_action_tensor.read()
			env_action = get_parent().env_action_tensor.read()
			action = agent_action[0]
			
			if env_action[0] == 1:
				$HUD._on_StartButton_pressed()
			
			if action == 0:
				velocity.x = 1
				velocity.y = 0
				
			if action == 1:
				velocity.x = -1
				velocity.y = 0
				
			if action == 2:
				velocity.y = 1
				velocity.x = 0
				
			if action == 3:
				velocity.y = -1
				velocity.x = 0
				
			speed_scale = get_parent().speed_scale
		else:	
			if Input.is_action_pressed("move_right"):
				velocity.x = 1
				velocity.y = 0
			if Input.is_action_pressed("move_left"):
				velocity.x = -1
				velocity.y = 0
			if Input.is_action_pressed("move_down"):
				velocity.y = 1
				velocity.x = 0
			if Input.is_action_pressed("move_up"):
				velocity.y = -1
				velocity.x = 0
	
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
		
		$Timer.start(0.0005)
		timeout = false
		get_parent().sem_physics.post()


func _on_Timer_timeout():
	get_parent()._observation_Function()


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
