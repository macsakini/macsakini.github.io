---
category: Linear Algebra
---

## What are Monte Carlo Simulation?

I know, I know, it looks like a guy's name. Well nope it isn't. Monte Carlo is a place. I always thought it Montreal. I wasn't so wrong. They kinda both start with M, and, hear me out, the both in Canada. Monte Carlo is a town known for its many casinos. So many, so so many and what is it about about casinos, there is a lot of betting. Now betting is super random. I mean, yes, there are odds, but they remain odds, they could blantantly go the other way. But something interesting is that even with the most random of stuff in our universe, there is still some sort of pattern. This pattern however, is hard to notice with a single or even few number of tries. 

Monte Carlo solves or comes close to solve this. Things we initially thought were random, turn out to have some sort of pattern due to the many times we will do the same task repetitively. This is called simulation and since we use monte carlo methods, its called monte carlo simulations. __MC sim__ for the cool math kids.

Even though the method was discovered long ago, the computational capabilities of the Pentium 😂 and earlier computers did not allow for such iterations. However, the M1, can do it in a flash. So if your are using Pentium, you need a makeover.

Lets do some examples:

# 1. Coin Tossing

I know its controversial why this would be a good example, since it has no real life use but trust me it makes the whole demonstration easy. So a coin toss is a bi thing. Not the LGBT kind, but a two-way thing, still not LGBT. You either get heads or tails, two way. As such we call it binomial. Each toss is also a single event, it literarily does not affect the next toss. These are __discrete events__. 

Assume we do ten tosses, what would be the probability of getting 3 or more heads. So we need to do the ten tosses and add up all the number of times it was head. Say we have 7 times, our proability is 7/10 or 70% or 0.7. Depending on where you stand.

Ten seems little however, why not repeat the tosses, maybe a fairly large amount of times, N, say 10000 times, will we get the same probability. R makes this super simple

```R
#Binomial Distribution
# First set the number of runs you would like
runs <- 10000

## One step will do a single round of toss 10 coins  and returns sum of the number of times the heads is > 3

one_step <- function(){
  sum(sample(c(0,1),10,replace = T)) > 3
}
#For Monte Carlo, we just replicate the same process, the total number of times as initiated. Note the replicate function.
# now we repeat that thing in 'runs' times.
montecarlo <- replicate(runs,one_step())
#this will print a lot of stuff but just note its trial you do not have to
print(montecarlo)

#to find the probability just sum all times it was true and divide by total runs
prob_toss <- sum(montecarlo)/runs

print(prob_toss)
```

Well, this about sums it up for the first part of Monte Carlo, for the next section we will do an example with another distribution. This one will involve continous data so make sure yopu read it.
If you liked, the read, give it a comment. Thank you.