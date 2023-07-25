# Design

A concurrent vector is represented as an array of lazily allocated buckets, of sizes `1, 2, 4 .. 2^63`:

```text
___________________________________
|  |    |        |                |
|  |    |        |                |
-----------------------------------
```

A buckets holds a number entries, as well as a lock to guard againt concurrent initialization:

```text
_____________
|  |  |  |  |
|  |  |  |  |
-------------
| UNLOCKED  |
-------------
```

An entry holds a slot for a value along with a flag indicating whether the slot is active:

```text
_____________________
| ACTIVE | INACTIVE |
|   42   |   NULL   |
---------------------
```

Writes acquire a unique index into the vector. The bucket holding the given entry is calculated
using the leading zeros instruction. If the bucket is already initialized, the value is written
to the slot, and the slot is marked as active. If the bucket has not been initialized, the thread
acquires the per-bucket initialization lock, allocates the bucket, and then writes the value. Note
that in the general case, writes are lock-free.

Reads use the same calculation to find the entry mapped to the given index, reading the value from
the slot if the flag indicates the slot is active. All reads are guaranteed to be lock-free.
