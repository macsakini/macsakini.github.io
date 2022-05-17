---
category: Data Structures
tags: dsa
classes: wide
---

Data Structures - Linked Lists, Arrays, Hashmap

This is a collection of some of the most fundamental data structures and functions done in Java.

Each file in this repository is a snippet of the java function assigned. The name of the repository is also closely related to the task at hand.
i.e getset.java concernes getters and setters.


## Linked Lists
```java
import java.util.Iterator;
import java.util.LinkedList;

class Linkylist {
    public static void main(String[] args){
        LinkedList<String> linky = new LinkedList<String>();

        //you can add elements to the LinkedAList 
        linky.add("Rob");
        linky.add("animal");
        linky.add("Hose");

        linky.getFirst();
        //linky.remove();

        System.out.println(linky);

        LinkedList<Integer> numlinky = new LinkedList<Integer>();

        numlinky.add(1);
        numlinky.add(78);
        numlinky.add(456);

        System.out.println(numlinky.get(2));

        //iterating through the linkedlist to find a number

        Iterator it = numlinky.iterator();

        while(it.hasNext()){
            if((int) it.next() == 78){
                System.out.println("We have found 78");
            }
        }
        

    }
}
```
## Arrays

```java
import java.util.ArrayList;
import java.util.List;

class arrayish {
    public static void main(String[] args){
        List<Integer> a = new ArrayList<Integer>();

        a.add(10);
        a.add(20);

        List <String> b = new ArrayList<String>();

        for(int i : a){
            System.out.println(String.valueOf(i));
            b.add(String.valueOf(i));
        }

        System.out.println(b);


    }
}
```
## HashMaps and HashTables

```java
import java.util.HashMap;

public class hashmapy {
    public static void main(String[] args){
        int a, b, c;
        a = 1;
        b = 2;
        c = 3;

        HashMap<String,Integer> happy = new HashMap<>();

        happy.put("a", a);
        happy.put("b", b);
        happy.put("c",c);


        System.out.println(happy.get("c"));
    })
 }
```

