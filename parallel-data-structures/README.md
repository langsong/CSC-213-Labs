# CSC 213 &ndash; Parallel Data Structures Lab

See complete details on the [lab webpage](http://www.cs.grinnell.edu/~curtsinger/teaching/2018S/CSC213/labs/parallel-data-structures.html).

Please complete your written responses to parts B and C below.

## Part B

### Invariants

* Invariant1:
For every value V that has been put into the queue p times and returned by
take q times, there must be p-q copies of this value on the queue. This
only holds if p >= q.

* Invariant2:
No value should ever be returned by take if it was not first passed to put
by some thread

* Invariant3:
If a thread puts value A and then takes value B, and no other thread puts
these specific values, A must be taken from the queue before putting B.

## Part C
*Briefly describe your implementation and synchronization strategy for the dictionary datatype here.*

We implement our dictionary by implementing a hashtable with a underlying bucket
linked list data structure. We lock individual bucket (linked list)
whenever we perform a set, contains, get or remove. So if two keys fall on
the same bucket, one of them needs to wait until the other finish the
operation. However, if two keys are not in the same bucket, they can
perform concurrent operations.

### CSC 207
*Which members of your group have taken or are currently taking CSC 207?*

Both of us have.

### Concurrent Accesses
*Describe which accesses to this data structure may proceed in parallel, and which accesses will block one another.*

So if two keys fall on the same bucket, one of them needs to wait until the
other finish the operation. However, if two keys are not in the same bucket, they can perform concurrent operations.

### Invariants
*Write your invariants for the dictionary data structure here.*

* Invariant1:
For every distinct key-value pair that has been set into the dict and
returned by get, we should get the same key-value pair.

* Invariant2:
Only -1 should be returned by get if the key was not first passed to set by some thread

* Invariant3:
Whenever after we set a key-value pair with the key that already exists in the
dict, when we call get, dict should return the updated value.
