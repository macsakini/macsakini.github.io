Algorithms - Merge Sort
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