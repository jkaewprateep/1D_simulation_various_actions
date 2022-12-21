"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/catcher.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

import ple
from ple import PLE

from ple.games import Catcher as catcher_game
from ple.games import base
from pygame.constants import K_a, K_d, K_h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left": K_a, "right": K_d }
nb_frames = 100000000000
global step
step = 0

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def	read_current_sate( string_gamestate ):
	global gameState

	gameState = p.getGameState()
	if string_gamestate in ['player_x', 'player_vel', 'fruit_x', 'fruit_y']:
		return round( gameState[string_gamestate], 1 )
	elif string_gamestate == 'close_gap':
		return round( pow( pow( gameState['fruit_x'], 2 ) + pow( gameState['fruit_y'], 2 ), 0.5 ), 1 )
	else:
		return None
		
def random_action( ): 
	global step
	step = step + 1

	player_x_value = read_current_sate('player_x')
	player_vel_value = read_current_sate('player_vel')
	fruit_x_value = read_current_sate('fruit_x')
	fruit_y_value = read_current_sate('fruit_y')
	close_gap_value = read_current_sate('close_gap')
	
	coeff_01 = player_x_value - ( 256 - fruit_y_value )
	coeff_02 = fruit_x_value - fruit_y_value
	coeff_03 = player_x_value - fruit_x_value
	
	control_matrix = tf.constant([ coeff_01, coeff_02, coeff_03 ], shape=(3, 1), dtype=tf.float32)
	
	## Optional ##
	# temp = tf.random.normal([1, 3], 1, 0.2, tf.float32)
	temp = tf.constant([ 0.1, 0.1, 0.1 ], shape=(3, 1), dtype=tf.float32 )
	## Optional ##
	# temp = tf.nn.softmax(temp[0])
	
	
	temp = tf.abs( tf.math.multiply( temp, control_matrix ) )
	print( "temp: " + str(temp) )
	action = int(tf.math.argmax(temp, axis=0))
	
	print( "action: " + str( action ) )

	return action

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = catcher_game(width=256, height=256)		
p = PLE(game_console, fps=30, display_screen=True)
p.init()

obs = p.getScreenRGB()	

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for i in range(nb_frames):
	
	if p.game_over():	
		step = 0

		p.init()
		p.reset_game()
		
	action = random_action( )
	reward = p.act(list(actions.values())[action])
