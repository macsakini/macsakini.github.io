Algorithms - Bubble Sort 
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