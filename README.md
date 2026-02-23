# Marathon-Pace-Predictor

Anyone who has ever ran a marathon before (myself included, I'm kind of a big deal) has estimated how fast they should run before they set off the start line. This is a straightforward process if you are planning a race between 1 and 13.1 miles. For those distances, it is very feasible to run over the distance you will be racing in training. This along with the consistency of your metabolic state during the race make it so that you can estimate easily off of pacing workouts, previous runs above that distance, and general feel. During a marathon, your metabolism is in greater flux, you (probably) haven't ran that distance or more in training, and pacing exercises have depreciated value because of the immense fatigue cost. 

This project helps solve that problem by using a model to predict marathon finish time/pace based off of prior training.

build notes:

could build predictor based on vickers data, then do probabalistic matching for finish times to append to afonseca dataset. this would take time though. probably out of scope for this week.

the idea rn is to build a prediction model based on vickers (they acheived a better prediction than riegel) and then use that to do probabalistic matching for the afonseca data. hopefully I will be able to train a better model on that, as it has more data with more training aswell.
