extends Node

var agent_action = [0.0]
var env_action = [0,0]

var sem_action
var sem_observation
var sem_physics

var mem
var env_action_tensor
var agent_action_tensor
var observation_tensor
var reward_tensor
var done_tensor

var policy
var policy_action

export(PackedScene) var mob_scene
export(PackedScene) var coin_scene
var score

var reward = 0
var done = 0
var speed_scale = 1.0
var timeout = true
var action = 0


func _ready():
	var args = Array(OS.get_cmdline_args())
	
	if args.has("t"):
		Engine.time_scale = 1.0;
		speed_scale = 1.0
	
	mem = cSharedMemory.new()
	sem_physics = Semaphore.new()
	sem_physics.post()
	if mem.exists():
		sem_action = cSharedMemorySemaphore.new()
		sem_observation = cSharedMemorySemaphore.new()

		sem_action.init("sem_action")
		sem_observation.init("sem_observation")
		
		agent_action_tensor = mem.findFloatTensor("agent_action")
		env_action_tensor = mem.findIntTensor("env_action")
		reward_tensor = mem.findFloatTensor("reward")
		observation_tensor = mem.findUintTensor("observation")
		done_tensor = mem.findIntTensor("done")
		print("Running as OpenAIGym environment")
		
		$Timer.start()
		#$HUD._on_StartButton_pressed()

	randomize()


func game_over():
	#print("func game_over()")
	#$StartTimer.stop()
	
	reward = -1.0
	done = 1.0

	#$ScoreTimer.stop()
	#$MobTimer.stop()
	#$HUD.show_game_over()
	#$Music.stop()
	#$DeathSound.play()
	
	#yield(get_tree().create_timer(4.0), "timeout")
	#new_game()
	#$HUD.hide_game_over()


func get_coin():
	reward = 1.0


func new_game():
	#print("func new_game()")
	done = 0
	reward = 0.0
	
	get_tree().call_group("mobs", "queue_free")
	get_tree().call_group("coins", "queue_free")
	
	score = 0
	$Player.start($StartPosition.position)
	$StartTimer.start()
	$HUD.update_score(score)
	$HUD.show_message("Get Ready")
	#$Music.play()
	
	$HUD.hide_game_over()
	

func _on_MobTimer_timeout():
	# Create a new instance of the Mob scene.
	var mob = mob_scene.instance()

	# Choose a random location on Path2D.
	var mob_spawn_location = get_node("MobPath/MobSpawnLocation")
	mob_spawn_location.offset = randi()

	# Set the mob's direction perpendicular to the path direction.
	var direction = mob_spawn_location.rotation + PI / 2

	# Set the mob's position to a random location.
	mob.position = mob_spawn_location.position

	# Add some randomness to the direction.
	direction += rand_range(-PI / 4, PI / 4)
	mob.rotation = direction

	# Choose the velocity for the mob.
	var velocity = Vector2(rand_range(150.0, 250.0), 0.0)
	mob.linear_velocity = velocity.rotated(direction)

	# Spawn the mob by adding it to the Main scene.
	add_child(mob)


func _on_CoinTimer_timeout():
	var coin = coin_scene.instance()

	#randomize()
	#var spawn_height = randi() % 600 + 1
	#var spawn_width = randi() % 600 + 1
	
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	var spawn_height = rng.randi_range(0, 800)
	var spawn_width = rng.randi_range(0, 1000)
	
	#print("spawn_height: ", spawn_height)
	#print("spawn_width: ", spawn_width)
	#print("")
	
	var position = Vector2(spawn_height, spawn_width)
	coin.position = position
	
	# Add some randomness to the direction.
	var direction = rand_range(-PI / 4, PI / 4)
	coin.rotation = direction

	# Choose the velocity for the mob.
	var velocity = Vector2(rand_range(0.0, 0.0), 0.0)
	coin.linear_velocity = velocity.rotated(direction)

	# Spawn the mob by adding it to the Main scene.
	add_child(coin)


func _on_ScoreTimer_timeout():
	score += 1
	$HUD.update_score(score)


func _on_StartTimer_timeout():
	$MobTimer.start()
	$CoinTimer.start()
	$ScoreTimer.start()
	$Timer.start()


func _get_screen_frame():
	# get data
	var img = get_viewport().get_texture().get_data()
	img.convert(4)
	img.resize(128, 128, 0)
	
	var height = img.get_height()
	var width = img.get_width()

	img.lock()
	var img_pool_vector = img.get_data()
	img.unlock()
	
	return img_pool_vector


func _on_Timer_timeout():
	#print("_on_Timer_timeout")
	sem_physics.wait()
	
	var return_values = _get_screen_frame()
	var observation = return_values
	
	if mem.exists():
		sem_action.wait()
		agent_action = agent_action_tensor.read()
		env_action = env_action_tensor.read()
		#print("agent_action: ", agent_action)
		#print("env_action: ", env_action)
		
		if env_action[0] == 1:
			$HUD._on_StartButton_pressed()
		
		#$Player.set_action(agent_action[0], speed_scale)
		action = agent_action[0]
		
		#print("mem.exists()\n")
		observation_tensor.write(observation)
		reward_tensor.write([reward])
		if reward == 1.0:
			reward = 0
		
		done_tensor.write([done])
		sem_observation.post()
		
	timeout = true
	sem_physics.post()
