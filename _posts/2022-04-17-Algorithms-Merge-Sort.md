---
category: Algorithms
tags: DSA, algo, sorting
classes: wide
---

## What is merge sort?

__Refer__ to https://www.coderstool.com/merge-sort for the original post.

Merge sort is one of the most efficient sorting algorithms. It works on the principle of Divide and Conquer. Merge sort repeatedly breaks down a list into several sublists until each sublist consists of a single element and merging those sublists in a manner that results into a sorted list.

Merge sort is a recursive algorithm that continually splits a list in half. If the list is empty or has one item, it is sorted by definition (the base case). If the list has more than one item, we split the list and recursively invoke a merge sort on both halves. Once the two halves are sorted, the fundamental operation, called a merge, is performed.

Merge sort is a sorting method that uses the divide and conquer method. It is one of the most respected algorithms, with a worst-case time complexity of O(n log n). Merge sort splits the array into equal halves before combining them in a sorted order.

## Algorithm

Merge sort divides the list into equal halves until it can't be divided any further.
If there is only one element in the list, it is sorted by definition.
Merge sort then joins the smaller sorted lists together, keeping the resultant list sorted as well.

- Step 1 − if it is only one element in the list it is already sorted, return.
- Step 2 − divide the list recursively into two halves until it can no more be divided.
- Step 3 − merge the smaller lists into new list in sorted order.


```java
import java.util.Arrays;

class mergesort {
    static int[] value = {14,33,27,19,42,44,78,6};
        
    static int n = value.length;
    public static void main(String[] args){
        mergesort(value, n);
    }

    public static void mergesort(int[] arr, int n){
        if(n == 1){
            return;
        }
        // System.out.println((n));

        // System.out.println(Arrays.toString(arr));

        int mid  = n/2;

        int[] l1 = new int[mid];
        
        int[] l2 = new int[n - mid];

        
        for (int i = 0; i < mid; i++) {
            l1[i] = arr[i];
        }

        for (int i = mid; i < n; i++) {
            l2[i - mid] = arr[i];
        }

        // System.out.println(Arrays.toString(l1));
        // System.out.println(Arrays.toString(l2));

        mergesort(l1, mid);

        mergesort(l2, n-mid);

        merge(l1,l2);


    }

    public static void merge(int[] a, int[] b){
        int[] c = new int[0];

        System.out.println(a.length);
        

        while(a.length > 0 && b.length > 0){
            if(a[0] > b[0]){
                c[c.length] = (b[0]);
                Arrays.asList(b).remove(b[0]);
            }else{
                c[c.length] = a[0];
                Arrays.asList(a).remove(a[0]);
            }
        }

        while(a.length > 0 ){
            c[c.length] = a[0];
            Arrays.asList(a).remove(a[0]);
        }

        while(b.length > 0 ){
            c[c.length] = (b[0]);
            Arrays.asList(b).remove(b[0]);
        }

        System.out.println(Arrays.toString(c));

    }
    
}
```


Divide via finding the range qqq of the position halfway between ppp and rrr. Do this step the same way we found the midpoint in binary search: add ppp and rrr, divide via 2, and round down.

Conquer with the aid of recursively sorting the subarrays in each of the two subproblems created by the divide step. That is, recursively type the subarray array[p..Q] and recursively sort the subarray array[q+1..R].

Combine by way of merging the two taken care of subarrays returned into the single sorted subarray array[p..R].

We need a base case. The base case is a subarray containing fewer than two elements, this is, while p geq rp≥rp, is greater than or same to, r, because a subarray and not using a factors or simply one detail is already looked after. So we will divide-conquer-combine most effective when p < rp<rp, is much less than, r.

Let's see an instance. Let's start with array conserving [14, 7, 3, 12, 9, 11, 6, 2], so that the first subarray is genuinely the whole array, array[0..7] (p=0p=0p, equals, 0 and r=7r=7r, equals, 7). This subarray has at the least elements, and so it is not a base case.

In the divide step, we compute q = 3q=3q, equals, 3.

The triumph over step has us kind the two subarrays array[0..3], which includes [14, 7, 3, 12], and array[4..7], which incorporates [9, 11, 6, 2]. When we come back from the triumph over step, every of the two subarrays is looked after: array[0..3] consists of [3, 7, 12, 14] and array[4..7] incorporates [2, 6, 9, 11], in order that the full array is [3, 7, 12, 14, 2, 6, 9, 11].

Finally, the combine step merges the two taken care of subarrays within the first 1/2 and the second one 1/2, generating the very last taken care of array [2, 3, 6, 7, 9, 11, 12, 14].

How did the subarray array[0..3] turn out to be sorted? The same manner. It has greater than factors, and so it is now not a base case. With p=0p=0p, equals, 0 and r=3r=3r, equals, three, compute q=1q=1q, equals, 1, recursively kind array[0..1] ([14, 7]) and array[2..3] ([3, 12]), ensuing in array[0..3] containing [7, 14, 3, 12], and merge the primary 1/2 with the second half of, producing [3, 7, 12, 14].

How did the subarray array[0..1] grow to be taken care of? With p=0p=0p, equals, zero and r=1r=1r, equals, 1, compute q=0q=0q, equals, 0, recursively type array[0..0] ([14]) and array[1..1] ([7]), resulting in array[0..1] still containing [14, 7], and merge the primary 1/2 with the second half, producing [7, 14].
