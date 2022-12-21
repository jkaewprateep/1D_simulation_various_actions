# 1D_simulation_various_actions
1D Simulation of randoms actions response with parameters for equations, we create some equations for Flappy birds games as 2D question now we do it for 1D and it will clearly explain of how the parallel parameters work at the remotes devices and IF-ELSE, RAND conditions within target devices without OS or Tensorflow using tranined parameters from Tensorflow as the Flappy bird games.

🧸💬 See it this way, he create guiding equation that perfrom well with the inputs as a data preparation step, you input anything into networks the AI try to learn as it process of optimizers, back-propagations or weights copy but it will take time as we also examines how does the optimizers work and their capabilities. It is not a magic tools create eveyrthing out from your inputs but you need to indicated and significants data or your experiences how you see the data. In this level the data engineer and business requirements need to work honestly on data input often they filled various inputs sometimes it is difficults to trackings such as one side abosulute. 💃( 👩‍🏫 )💬 You see all are correct and the optimizer perfrom normal as normal process but there are more than critical points.

## Catcher games ###

Basic information for computer visions and games related, they are using X and Y axises but Y is inverse order then near start from top and far is the end of the monitor screen, you know that by extension monitors too. 🦹💬 Not only try catchers games but extended monitors will not have a problem with refresh rates.


### Axises and Dimensions ###

Change of X is player_x_value and the change of Y is ( 256 - fruit_y_value ) from the game getGameState() in PLE. Our objective is to reduces the contrast between ```player_x_value and player_x_value``` to have a rewards return. 🐑💬 For AI networks training it will select the most functions response, waiting the AI to learn from rewards you need a galics hit your bar but ```fruit_x_value - fruit_y_value``` tell the AI to start move bar and the AI will ignored because the different value or scales different, acclerate functions. The function ```SoftMax()``` can apply for significant value contrast and ```Random()``` functions also.

```
_____________________________________________________________________
         dX
|------------------|🧄💦
                  _
                  |
                  | dY
                  _
                    🐶💦💦
_____________________________________________________________________
```

### Our keys equations ###

The coeff_01 is initail telling one parameter AI can controls is ```player_x_value``` byside the equation ```fruit_x_value - fruit_y_value``` that create the AI need to move the bar in X axis, target functions is ```player_x_value - fruit_x_value``` because all conditions is complete as the ```player_x_value``` change and ```fruit_x_value``` values does not change. The sameple result value is ```{ 7.2, 12.4, 6.0  } ==> action: 1``` and ```{ 7.2,   2.2, 4.2 } ==> action: 0```

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
