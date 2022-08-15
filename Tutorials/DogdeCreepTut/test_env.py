import gym
import numpy as np
import random
import matplotlib.pylab as plt
import cv2
import io 
import time

from dodgeCreep import dodgeCreepEnv

if __name__ == '__main__':
	num_warmup = 1000
	num_train = 200000
	num_eval = 0
	buffer_length = 600000

	GODOT_BIN_PATH = "dodge_the_creeps/DodgeCreep.x86_64"
	env_abs_path = "dodge_the_creeps/DodgeCreep.pck"
	env = dodgeCreepEnv(exec_path=GODOT_BIN_PATH, env_path=env_abs_path, turbo_mode=True)

	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	#print("num_actions: ", num_actions)

	for episode in range(1000):
		print("episode: ", episode)

		state = env.reset()
		step = 0
		while True:
			action = random.randint(0,4)
			#action = 3
			#print("action: ", action)

			state_next, reward, done, _ = env.step(action)
			#state_next = state_next.detach().numpy()
			#print("state_next: ", state_next)
			#print("reward: ", reward)
			#print("done: ", done)

			state_next = np.reshape(state_next, (128,128,3)) / 255.0
			state_next = np.array(state_next).astype(np.float32)
			state_next = cv2.cvtColor(state_next, cv2.COLOR_BGR2RGB)
			#state_next = state_next.astype(np.uint8)
			#state_next = cv2.resize(state_next, (80, 80), interpolation=cv2.INTER_CUBIC)
			#state_next = 0.299*state_next[:,:,0] + 0.587*state_next[:,:,1] + 0.114*state_next[:,:,2]

			# convert everything to black and white (agent will train faster)
			#state_next[state_next < 70] = 0
			#state_next[state_next >= 100] = 255

			#state_next = np.array(state_next).astype(np.float32) / 255.0
			#state_next = state_next / 255.0

			cv2.imshow("state_next: ", state_next)
			if cv2.waitKey(25) & 0xFF == ord("q"):
				cv2.destroyAllWindows()

			#print("")
			#time.sleep(0.1)
			step += 1

			if done == True:
				print("step: ", step)
				break
		
	env.close()
