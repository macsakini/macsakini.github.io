Algorithms - Select Sort
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