---
category: Linear Algebra
---
Now, lets talk about something else apart from the binomial distribution. Say you have a circle, and you want to drop balls over it. One by one you drop a ball over the circle, say dish. The question is how many balls are gonna go inside it.

As a mathematician, anything that involves circles or curves or anything circul-ish should bring to your mind the symbol $$ \pi $$. PI is a lot of things but just understanding that somehow a circle would be involved makes the whole thing wrapped up. Next the field around the circle most cases is usually a square. Look at the diagram below.

Notice how the balls might fall out of the circle and into that area. Would it be right if i said, the proability of all balls being in the circle is $$ \pi/4 $$

Well, we will see. This is a mathematical standard but a simulation might better prove this. Hope your RStudio is open.

```R
#Set the number of runs
runs <- 5000

#drop one ball using the function drop_ball
drop_ball <- function(){
    #simulate the x-coordinate of the ball
    x <- runif(n = 1, min = 1e-12, max = .9999999999)
  
    #simulate the y-coordinate of the ball
    y <- runif(n = 1, min = 1e-12, max = .9999999999)
    
    #We are using the uniform distribution between 0 and 1

    # It is a mathematical constant that anything that fulfills the rule below is a coordinate in the unit circle.
    # return this as the answer, is it true or not
    (x**2 + y**2) < 1
}

mc <- sum(replicate(runs, drop_ball))/runs

#print the proportion
print(mc)
```

The answer printed above should fulfil the earlier rule that i mentioned of the Pr(a) coverging towards pi/4.

Monte Carlo has endless uses but a big part about simulations is knowing what you need to achieve. This will guide you on the right path and actions. It can be used for more versatile uses: predicting risk of a volatile security or stock, predicting number of items that will go to SCRAP in a conveyor chain and so many. Particle physics uses monte carlo a lot, i mean they dont have unlimited supply of __particles__ 😬.

This about sums up the Monte Carlo series. However for any queries, am a comet away. (Get it, coz comment is like comet🥲. Bet you didnt see that coming, ditchya)