---
category: Data Structures
tags: dsa
---

Data Structures - Stacks and Queues in Java.

## Stacks
You are probably wondering, what are stacks. Well Stacks are stacks 😛. It is in the name right. Well a stack is how you put your plates. Or how you play that cup game. You just put things over each other. 

The main principle behind stacks is LIFO: Last In First Out. 
The main two actions a stack performs is pushing and poping. You push something into the stack and you pop something out of the stack.

Java has a built in stack framework. Find below an image showcasing this characteristic.

__Image Goes HERE....__

Java implements it quite nicely, create a new stack called games. To add items into the games stack just use the add function. To remove stuff out of the games stack just use pop.

The second example shows a character stack list. Here am justing adding single characters such as 'A', 'B' and so on.
```java
import java.util.Stack;

public class stacky {
    public static void main(String[] args){
        Stack<String> games = new Stack<String>();

        games. add("Call of Duty");

        games.add("Super Monkey Ball");

        games.add("Guitar Hero");

        System.out.println(games);

        //if you would like to get topmost one 
        System.out.println(games.pop());

        //to get second from top
        System.out.println(games.pop());

        // to get remaining stack after two pop operations
        System.out.println(games);

        //to check the first item on the stack 
        System.out.println(games.peek());

        // Y
        // B
        // R

        Stack<Character> tower = new Stack<Character>();

        tower.add('R');
        tower.add('B');
        tower. add('R');

        System.out.println(tower.empty());
        System.out.println(tower.get(1));
        System.out.println(tower.set(1, 'P'));
        System.out.println(tower.size());



    }
}
```

The queue is just stacks wacky brother. Same concept except the queue is an actual groceries shop queue or the line at the hospital. So whoever walked in first, gets served first. This is an interesting principle called FIFO: First In First Out. 

The actions that describe a queue at best are enqueue and dequeue. However, Java makes it simple, you add or poll a queue. Make sure to copy paste the code and try it out.

__Image does here__

## Queues
```java
import java.util.LinkedList;
import java.util.Queue;

public class queueish {
    public static void main(String[] args){
        //Example one
        Queue<String> bbq = new LinkedList<String>();

        bbq.add("Jeff");

        bbq.add("Tyrique");

        bbq.add("Susan");
        
        bbq.peek();

        System.err.println(bbq);

        //Jeff
        bbq.poll();
        //Tyrqique
        bbq.poll();
        //Susan
        System.out.println(bbq.poll());

        //Example two

        Queue<String> q = new LinkedList<String>();

        q.add("A");
        q.add("B");
        q.add("C");

        q.poll();

        System.out.println(q);

        System.out.println(q.size());

        System.out.println(q.contains("A"));

        System.out.println(q.toArray()[1]);

    }

}
```