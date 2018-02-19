### Semantic-Segmentation
André Marais
2018-02-19
---


##### Reflections
Once again, the project walkthrough really helped to get things up and running. I used the architecture discussed during the classes, but I also added a kernel regularizer  at every convolution. This really
improved the accuracy of the model. I also added image augmentation to improve the accuracy. The augmentation flipped the image along the x- and y-axis randomly and then also changed the brightness.

The challenge I faced was to improve the accuracy enough so that the model does not 'spill over' with its road classifications. I also tried to detect the other road, but it didn't seem to work well enough.

 I'm still battling to connect to my EC2 instance, so I used FloydHub for my model training and classification. It's quite cheap - $12 for 10 hours of a Tesla K80 :) But it adds up... I wanted to run my model on
  a video, but unfortunately the first stab at it was really bad and I didn't have time to figure out why.


#### Challenges faced
Naturally, just to get the model to run. Neural networks are tricky to set up, but I realized that it gets easier the more time you spend with it. I also battled to add another layer to classify.
I think I managed to get the syntax right, but the training and classification still fails.
