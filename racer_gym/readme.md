# Race Car Gymnasium

from the docs: 

## Description
The easiest control task to learn from pixels - a top-down racing environment. The generated track is random every episode. <br/>
Some indicators are shown at the bottom of the window along with the state RGB buffer. <br/>
From left to right: true speed, four ABS sensors, steering wheel position, and gyroscope. To play yourself (it’s rather fast for humans), type:<br/>
```python main.py --play```


## Action Space
If continuous there are 3 actions :
    0: steering, -1 is full left, +1 is full right
    1: gas
    2: braking

If discrete there are 5 actions:<br/>
    0: do nothing<br/>
    1: steer right<br/>
    2: steer left<br/>
    3: gas<br/>
    4: brake<br/>

## Observation Space
A top-down 96x96 RGB image of the car and race track.

## Rewards
The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

## Starting State
The car starts at rest in the center of the road.

## Episode Termination
The episode finishes when all the tiles are visited. The car can also go outside the playfield - that is, far off the track, in which case it will receive -100 reward and die.




