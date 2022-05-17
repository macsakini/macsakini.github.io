---
category: Algorithms
tags: dsa
classes: wide
---

## What is bubble sort?

Bubble sort simply swaps adjacent items if the two it is looking at are incorrectly ordered. It continues moving down the list, and will keep on going until everything is sorted.

Now, let's go over the code you wrote up.

This will start a index 'a' pointing to the first element of the list. The inner loop creates a second index which goes to the last element (size - 1) and will decrease it until it reaches 'a'. The first step will be to compare the last element with the second to last element, and swap them if needed. 'b' is then decremented and the second from last and third from last are compared. This continues until the first and second element in the list are compared. Once 'b' has reached it's termination condition we increment 'a' and keep going. This increment of 'a' is possible since we know that the smallest element in the list will be correctly bubbled to the first element so there would be no need for 'b' to check values less than 'a'.

Here's a small example to illustrate the idea (underlined number is 'a', italicized number is b, bolded numbers are two being compared)

## Algorithm

Unsorted List:
4 6 2 1 3 5

First Pass:
4 6 2 1 3 5 -- initially 'a' is pointing to the first element, 'b' to the last
4 6 2 1 5 3 -- the last two elements are swapped
4 6 2 1 5 3 -- 'b' is decremented and two new elements are compared
4 6 2 1 5 3 -- no swap is made and 'b' is decremented again
4 6 1 2 5 3 -- next two have been swapped
4 6 1 2 5 3 -- 'b' is decremented again
4 1 6 2 5 3 -- another swap is made
4 1 6 2 5 3 -- 'b' is decremented
1 4 6 2 5 3 -- swap is made and 'b' cannot be decremented any more
1 4 6 2 5 3 -- 'a' is incremented and 'b' is reset to the end
...
1 2 4 6 3 5 -- After second pass
...
1 2 3 4 6 5 -- After third pass
...
1 2 3 4 5 6 -- After forth pass

You can see in this short example that we can see how the bubble sort will 'bubble' the smallest element up to the front and be sure that the elements less than 'a' are fine.

```java
import java.util.Arrays;

class BubbleSort{
    //static ArrayList<Integer> value = new ArrayList<Integer>(Arrays.asList(14,33,27,35,10));
    static int[] value = {14,33,27,35,10};
    
    public static void main(String[] args){

        System.out.println(Arrays.toString(value));
        for(int i = 0; i < (value.length - 1); i++){
            Boolean swapped = false;
            for(int j = 0; j < (value.length - 1); j++){
                System.out.println(j);
                if(value[j] > value[j+1]){
                    swap(j, j+1);
                    swapped = true;
                    System.out.println(Arrays.toString(value));
                }
            }
            System.out.println(swapped);
            if(swapped == false){
                break;
            }
        }

        System.out.println(Arrays.toString(value));
    }

    public static String swap(int i, int j){
        int temp = value[i];
        value[i] = value[j];
        value[j] = temp;
        return"success";
    }
}
```