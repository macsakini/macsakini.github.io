---
category: Linear Algebra
---
Linear Algebra Series 2 - Linear Independence and Dependence

I will try to avoid math, so test 1, 

$$ A \implies B \, \exists \, x \: \in \R $$

And yes, it is working. This was just a test, dont copy that. Now lets get down to it. 
First, order of business, is linear __independence__. This is a major concept in linear algebra but one the __Brits__ seem to never understand 🤨. I'll describe both the formal and you versions for better understanding.

## Formal Definition.

__Linear independence__. Say we have set of vectors $$ x_1, x_2 \land x_3 $$ in a linear subspace, they can only be termed independent if $$ \exists \, c_1, c_2, c_3 $$ such that: 

$$ c_1 * x_1 + c_2 * x_2 + c_3 * x_3 = 0 $$

$$ \land \: \therefore $$ a null vector, where $$ c_1 = c_2 = c_3 = 0 $$

__Linear dependence__ on the other hand is the same except atleast one scalar is non-zero.

__Note__ Linearly dependent vectors can be written as a linear combination of linearly independent vectors.

## Logical Definition.