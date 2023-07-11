extends Node2D

var mem:cSharedMemory
var sem_action:cSharedMemorySemaphore
var sem_observation:cSharedMemorySemaphore
var sem_environment:cSharedMemorySemaphore
var tensor_action:cPersistentFloatTensor
var tensor_observation:cPersistentFloatTensor
var tensor_env:cPersistentIntTensor
var agent_action = [1.]
var observation = [1.]

var time = 0.0
var deltat = 0.1

func _ready():
	mem = cSharedMemory.new()
	mem.init("environment")
	if mem.exists():
#		push_error('Initializing')
#		sem_action = cSharedMemorySemaphore.new()
#		sem_observation = cSharedMemorySemaphore.new()
#		sem_environment = cSharedMemorySemaphore.new()
#		sem_action.init("sem_action")
#		sem_observation.init("sem_observation")
#		sem_environment.init("sem_environment")
		
		#Shared memory tensors
#		tensor_action = mem.findFloatTensor("agent_action")
#		tensor_observation = mem.findFloatTensor("observation")
#		sem_environment.post()
		push_error('Done init')
	else:
		push_error('No shared memory detected')

func _process(delta):
	if mem.exists():
#		$Output.text = 'Waiting action'
#		push_error('Waiting action')
#		sem_action.post()
		time += 1
#		agent_action = tensor_action.read()
#		push_error('Writing observation')
#		$Output.text = 'Writing observation'
#		tensor_observation.write(observation)
#		sem_observation.post()
		$Output.text = 'Time: ' + str(time)
#		time = 0.0
	
