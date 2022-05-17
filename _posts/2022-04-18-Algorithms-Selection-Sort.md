---
category: Algorithms
tags: dsa
classes: wide
---

## What is selection sort?

Selection sort algorithm picks the minimum and swaps it with the element at current position.

Suppose the array is:
```python
[5 2 3 8 4 5 6] 
```

## Algorithm

Let's distinguish the two 5's as 5(a) and 5(b) .

So our array is:
```python
[5(a) 3 4 5(b) 2 6 8]
```

After iteration 1:
- 2 will be swapped with the element in 1st position:

So our array becomes:
```python
[2 3 4 5(b) 5(a) 6 8]
```

Since now our array is in sorted order and we clearly see that 5(a) comes before 5(b) in initial array but not in the sorted array.

```java

import java.util.Arrays;

class selectsort {
    static int[] value = {14,33,27,245,10,35,19,42,44,78,6};
    public static void main(String[] args){
        int n = value.length;

        System.out.println(Arrays.toString(value));
        
        for(int i = 0; i < (n - 1); i++){
            int min = i;

            for(int j = (i + 1); j < n; j++){
                System.out.println(Arrays.toString(value));

                if (value[j] < value[min]) {
                    min = j;
                }
            }
            System.out.println(Arrays.toString(value));

            if (min != i){
                swap(min, i);
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