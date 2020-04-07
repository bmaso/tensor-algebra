# Tensor Algebra

A Scala language implementation of a tensor algebra. The basic operations of
the algebra are both highly expressive, and highly parallelizable.

The basic operations of the algebra are individually simple to understand. The
algebra is expressive, meaning that even extremely complex N-dimensional computations
can be completely expressed as combinations of these basic operations. For example,
 matrix multiplication, convolution and cross-correlation, Fourier
transforms, etc. can all be expressed with this algebra without looping or recursion.

Each of the algebra's operations is amenable to highly parallelized implementations.
Therefore it is possible to define different evaluators for efficiently calculating
tensor algebra expressions with parallel processors of
varying scale:
* Hardware computational elements such as GPUs or GPU arrays, multi-core CPUS,
  DSPs, FPGAs, etc.
* Cloud-based map-reduce systems such as Spark, Google Tensorflow, etc.

## Inspiration

Consider the *convolution* and *cross-correlation* operations. These are common
operations in image and audio processing, neural networks, and several other
real-world computational scenarios. Both operations combine two N-dimensional
arrays (aka tensors) into a new N-dimensional array of the same dimension.
If you are unfamiliar, read [this good explanation of 2-D convolution and
cross-correlation in medical imaging](https://glassboxmedicine.com/2019/07/26/convolution-vs-cross-correlation/).

Imperative and functional implementations of these operations are fairly obvious.

* An imperative implementation requires nested looping over the indexes of both
  tensors, placing the results in a result tensor. The principal feature
  of this implementation is nested `for` loops.
* A functional implementation abstracts the looping with mapping and folding. The
  principal feature of this implementation is use of nested `map` and `fold` operations
  on the tensor element values.

The imperative implementation requires the implementor to code the execution
plan for the operations explicitly. In different execution environments, e.g. CPU vs
GPU, will require very different implementations.

The functional implementation abstracts the looping aspects of the implementation,
allowing the data container (the tensor classes/objects) to "decide" the best
way to carry out the calculation at runtime. Heavy use of recursion makes the runtime's
job of devising an efficient execution plan quite difficult -- impractically so
for even moderately complex tensor expressions such as N-dimensional convolution.

### Convolution and Cross-Correlation Implemented in Tensor Algebra

A third option exists for implementing arbitrarily complex operations like N-dimensional
convolution and cross-correlation. By combing a few simple operations (that is,
basic tensor algebra) we can define these and even more complex operations such as
N-dimensional Fourier transforms. The basic operations of this algebra are:
* *translate* -- define a new tensor by "moving" values in an original
  tensor to different indexes
* *permute* -- This is the general name for rotating a tensor, allowing rotation
  in multiple dimensions at once
* *reshape* -- Without altering the overall number of elements in a tensor, define
  a tensor which takes an original tensor's elements while altering the dimensionality
  and arity
* *split* -- divide a tensor into several tensors of the same
  dimensionality by partitioning or segmenting along one or more dimensions
* *join* -- additively "stacking" multiple tensors of the same dimensionality
  into a single tensor, which will also have the same dimensionality
* *broadcast* -- Projecting a lower dimensionality tensor into a higher
  dimensionality by duplication
* *reduce* -- Creating a new tensor with reduced dimensionality from an original
  by combining elements along one or more dimensions using an arbitrary associative
  arithmetic operation.

Defining a tensor algebra utilizing only these basic operations it is possible
to define very complex operations from linear and high-dimensional algebras including
matrix multiplication, convolution, cross-correlation, Fourier transformation, and
many more.

By using a tensor algebra comprised of highly parallelizable basic operations,
we can define different tensor expression evaluators which will automatically
plan very efficient use of highly parallelized computing resources, such as GPAs,
DSPs, FGPAs, multi-core CPUs, as well as cloud-based map-reduce systems.

#### Algebraic Definitions of Convolution and Cross-Correlation

## Some Simple Tensor Operations

Definitions of some basic operations that one would expect to be present in
an N-dimensional numerical array algebra:
* Scalar values
* Construction of a new tensor of arbitrary arity and dimension
  * Constructing a tensor filled with zeros or ones
  * Constructing an identity matrix
* Arbitrary associative arithmetic combination of 2 or more tensors
  * Aka the *acomb* operation

### Scalar Values

### Constructing Tensors

### *Acomb* "Arbitrary Associative Arithmetic Combination"

## More Rigorous Definitions of The Basic Algebra Operations

### *Translate*

### *Permute*

### *Reshape*

### *Split*

### *Join*

### *Broadcast*

### *Reduce*
