# 1D_simulation_various_actions
1D Simulation of randoms actions response with parameters for equations, we create some equations for Flappy birds games as 2D question now we do it for 1D and it will clearly explain of how the parallel parameters work at the remotes devices and IF-ELSE, RAND conditions within target devices without OS or Tensorflow using tranined parameters from Tensorflow as the Flappy bird games.


## Catcher games ###

```
coeff_01 = player_x_value - ( 256 - fruit_y_value )
coeff_02 = fruit_x_value - fruit_y_value
coeff_03 = player_x_value - fruit_x_value

control_matrix = tf.constant([ coeff_01, coeff_02, coeff_03 ], shape=(3, 1), dtype=tf.float32)
temp = tf.random.normal([3, 1], 1, 0.2, tf.float32)
temp = tf.abs( tf.math.multiply( temp, control_matrix ) )
print( "temp: " + str(temp) )

action = int(tf.math.argmax(temp, axis=0))
print( "action: " + str( action ) )
```



## Result image ##

#### Catcher games ####

![Alt text](https://github.com/jkaewprateep/1D_simulation_various_actions/blob/main/random_catcher.gif?raw=true "Title")


#### Flappy Bird games ###

![Alt text](https://github.com/jkaewprateep/1D_simulation_various_actions/blob/main/FlappyBirds.gif?raw=true "Title")


#### Pixel Copter games ###

![Alt text](https://github.com/jkaewprateep/1D_simulation_various_actions/blob/main/random_copter.gif?raw=true "Title")
