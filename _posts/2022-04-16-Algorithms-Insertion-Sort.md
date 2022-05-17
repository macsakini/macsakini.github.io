---
category: Algorithms
tags: dsa
classes: wide
---

## What is insertion sort?

Algorithm is stable and does not use the extra auxiliary space.

For small sized unsorted array it is the best sorting algorithm.

It is part of sort and introsort.

Time complexity

Best case: $$ \theta(n) $$

Worst case: $$ \theta(n^2) $$

## Algorithm

Step 1: Outer loop is run from 1 to len(array)

Step 2: In inner loop the current and next element are compared and if the next element is smaller than the current element positions change

Step 3: Once the swapping is done those elements becomes sorted and next element is compared with these sorted subarray and step 2 is repeated.

Step 4: Step 2 and Step 3 are taking place in the inner loop

Step 5: Elements are sorted

Some key points with respect to insertion sort

```java
import java.util.Arrays;

class insertionsort {
    static int[] value = {14,33,27,35,10};
    public static void main(String[] args){
        int holePosition;
        int valueToInsert;

        for(int i = 1; i < value.length; i++ ){
            valueToInsert = value[i];
            
            holePosition = i;

            while(holePosition > 0 && value[holePosition - 1] > valueToInsert){
                
                value[holePosition] = value[holePosition - 1];
                
                holePosition = holePosition - 1;

                System.out.println(Arrays.toString(value));
                
            }

            value[holePosition] = valueToInsert;
        }

        System.out.println(Arrays.toString(value));
    }    
}
```