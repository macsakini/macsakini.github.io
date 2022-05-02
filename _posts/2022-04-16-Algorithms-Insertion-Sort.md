Algorithms - Insertion Sort
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