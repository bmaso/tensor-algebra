# Tensor Algebra

A Scala language implementation of a tensor algebra. The basic operations of
the algebra are both highly expressive, and highly parallelizable.

Each of the algebra's operations is amenable to highly parallelized implementations.
Therefore it is possible to define different evaluators for efficiently calculating
tensor algebra expressions with parallel processors of
varying scale:
* Hardware computational elements such as GPUs or GPU arrays, multi-core CPUS,
  DSPs, FPGAs, etc.
* Cloud-based map-reduce systems such as Spark, Google Tensorflow, etc.

The basic operations of the algebra are individually simple to understand. The
algebra is expressive, meaning that even extremely complex N-dimensional computations
can be completely expressed as combinations of these basic operations. For example,
 matrix multiplication, convolution and cross-correlation, Fourier
transforms, etc. can all be expressed with this algebra without looping or recursion.

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

The conclusion: in Scala processing of numerical data, especially multi-dimensional numerical data, is easy to express but rather slow. Mechanisms for speeding up either imperative or functional expressions of numerical data are lacking. Tensor algebra promises to fix these issues.   

### Tensor Algebra's Basic Operations

A third option exists for expressing and efficiently evaluating arbitrarily complex numerical operations, such as N-dimensional
convolution and cross-correlation, in Scala and the JVM in general. This tensor algebra utilizes a few simple operations to support expressing complex computations in a way that is relatively easy to evaluate efficiently using highly parallel processing.

Recall that a "tensor" is just the formal name for an N-dimensional rectangular array of any dimension. This includes 0-D (scalar values), 1-D (arrays), 2-D (matrixes), 3-D (rectangular volumes), and higher.

The basic operations of this tensor algebra are:
* ***translate*** -- define a new tensor by "moving" values in an original
  tensor to different indexes using fixed offset in each dimension
* ***permute*** -- This is the general term for rotating a tensor, allowing rotation in multiple dimensions at once
* ***reshape*** -- Without altering the overall number of elements in a tensor, define
  a tensor which takes an original tensor's elements while altering the dimensionality and arity.
  For example changing a 3x4x5 tensor into a 10x6 tensor. Same number of elements,
  different dimensionality and/or arity.
* ***split*** -- divide a tensor into several tensors of reduced
  dimensionality along one or more dimensions
* ***join*** -- additively "stacking" multiple tensors of the same dimensionality
  into a single tensor of higher dimensionality
* ***broadcast*** -- Projecting a lower dimensionality tensor into a higher
  dimensionality one by duplication
* ***reduce*** -- Creating a new tensor with reduced dimensionality from an original
  by combining elements along one or more dimensions using an arbitrary associative arithmetic operation.
  * An arithmetic function that is both commutative & associative yields a much greater opportunity for evaluator parallelism

With this tensor algebra we define very complex operations from linear and high-dimensional algebras including
matrix multiplication, convolution, cross-correlation, Fourier transformation, complex image and sound manipulation functions. By interpreting data graphs as sparse matrices, you can implement many complex
graph algorithms in the tensor algebra as well.

### Example: 2-D Cross-Correlation Defined using Tensor Algebra

Let's consider the cross-correlation of an ![NxM][NxM] 2-D matrix ![Mat][Mat] with a ![3x3][3x3] 2-D matrix ![k][k] (aka *kernel*). The ![star][star] symbol is usually used to represent cross-correlation in
standard math.

![Eq1][Eq1]

Each element ![Cij][Cij] of the cross-correlation is the sum of the element-wise product of 9 elements in the neighborhood of ![Matij][Matij] with the 9 elements of ![k][k]. Our goal is to define the ![star][star] operation using tensor algebra operations only. First, let's define ![star][star] in standard algebra with the equation:

![Eq2][Eq2]

 where ![Matstar][Matstar] is a "dilated" version of ![Mat][Mat]: expanded by 1 element in each 2-D direction (intuitively: up, down, left, & right) with 0 values along the edges and in the corners.

#### "![star][star]" Operator as a 4-D ***reduce*** in the Tensor Algebra

In the tensor algebra you do not use iterative operators such as "![sigma][sigma]". Instead we define all arithmetic combinations of tensors using ***reduce***. We define the 2-D "![star][star]" operation combining an ![NxM][NxM] shaped tensor with a ![3x3][3x3] shaped kernel tensor as a single ***reduce*** of a *4-D* ![NxMx9x2][NxMx9x2] shaped tensor back down to 2-D dimensionality.

We can define this in 5-steps using tensor algebra operations:

1. Construct 9 translated versions of ![Mat][Mat], and join them into a single ![NxMx9][NxMx9] shaped tensor
  * One version each translated with offsets *(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1),
    (1, 0), and (1, 1)*
  * The values of the 9 tensors at index *(i, j)* taken together are the values of the original ![3x3][3x3]
    neighborhood of element *(i, j)* in ![Mat][Mat]
2. Broadcast ![k][k] into a ![NxMx3x3][NxMx3x3] shaped tensor, and reshape to ![NxMx9][NxMx9] shape
3. Join tensors from steps (1) and (2) into a single ![NxMx9x2][NxMx9x2] shaped tensor
4. Reduce the tensor from (3) down to ![NxMx9][NxMx9] shape by multiplicative reduction of the 4*th* dimension
  * In effect, this step multiplies each element in the neighborhood of each *i,j-th* ![Mat][Mat]
    element with the corresponding kernel element.
5. Reduce the ![NxMx9][NxMx9] tensor from (4) down to ![NxM][NxM] shape by additive reduction of the 3*rd* dimension
  * In effect, this sums together the 9 products of the original ![Mat][Mat] elements and the
    kernel elements in the neighborhood of each *i,j-th* element of ![Mat][Mat]

(Note: steps (4) and (5) would more efficiently be defined as a single reduce along both 4*th* and 3*rd* dimensions. For simplicity, the steps are described explicitly to make relationship to the ![Cij][Cij] more obvious.)

As a single equation using tensor algebra:

## Why is This better than Imperative or Functional Programming?

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

In the tensor algebra a scalar value is nothing more than a 0-dimension  tensor.

### Constructing Tensors

### *Carith-op* "Commutative & Associative Arithmetic Operation"

## More Rigorous Definitions of The Basic Algebra Operations

### *Translate*

### *Permute*

### *Reshape*

### *Split*

### *Join*

### *Broadcast*

### *Reduce*


[NxM]: http://www.sciweavers.org/tex2img.php?eq=N\times%20M&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[3x3]: http://www.sciweavers.org/tex2img.php?eq=3\times3&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[NxMx9]: http://www.sciweavers.org/tex2img.php?eq=N\times%20M\times9&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[NxMx3x3]: http://www.sciweavers.org/tex2img.php?eq=N\times%20M\times3\times3&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[NxMx9x2]: http://www.sciweavers.org/tex2img.php?eq=N\times%20M\times9\times2&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[Mat]: http://www.sciweavers.org/tex2img.php?eq=Mat&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[Matstar]: http://www.sciweavers.org/tex2img.php?eq=Mat^*&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[k]: http://www.sciweavers.org/tex2img.php?eq=k&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[C]: http://www.sciweavers.org/tex2img.php?eq=C&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[Eq1]: http://www.sciweavers.org/tex2img.php?eq=C=Mat%20\star%20k&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[Cij]: http://www.sciweavers.org/tex2img.php?eq=C_{ij}&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[star]: http://www.sciweavers.org/tex2img.php?eq=\star&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[sigma]: http://www.sciweavers.org/tex2img.php?eq=\sum&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[Matij]: http://www.sciweavers.org/tex2img.php?eq=Mat_{ij}&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[Eq2]: http://www.sciweavers.org/tex2img.php?eq=C_{ij}=\sum_{s=-1}^1\sum_{t=-1}^1%20Mat^*_{(i%2Bs)(j%2Bt)}k_{st}&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[Mat*_def]: http://www.sciweavers.org/tex2img.php?eq=Mat^*=join_2(join_1(translate^2([1,1],0,Mat),broadcast^2([1,M%2b1],0)),broadcast^2([N%2b2,1],0))&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0
[irange1]: http://www.sciweavers.org/tex2img.php?eq=\{1,N%2B1\}&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[jrange1]: http://www.sciweavers.org/tex2img.php?eq=\{1,M%2B1\}&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
[N+2xM+2]: http://www.sciweavers.org/tex2img.php?eq=(N%2B2)\times%20(M%2B2)&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=
