package bmaso.tensoralg.abstractions

import cats.free.Free

/**
 * Each evaluator defines its own concrete tensor type, and defines its operations
 * in terms of this tensor type.
 **/
trait TensorAlgebra {
  type Tensor <: bmaso.tensoralg.abstractions.Tensor

  /**
   * The tensor algebraic expressions which are common to all concrete tensor algebras.
   * Note what additional expression types must be added to concrete subtypes:
   * * Expressions for constructing tensors from Scala data
   * * Expressions for retrieving data elements from tensors
   * * ***reduce* expressions
   **/
  trait TensorExprOp[T]

  /**
   * Represents the translation of a tensor in zero or more dimensions.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `translate` to perform invariant checks.
   **/
  case class Translate(tensor: this.Tensor, offsets: Array[Long]) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the expansion of a tensor in one or more dimensions bby duplication.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `broadcast` to perform invariant checks.
   **/
  case class Broadcast(tensor: this.Tensor, baseMagnitude: Array[Long]) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the re-indexing of a tensor: this changes the dimensionality without
   * changing the `elementSize`.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `reshape` to perform invariant checks.
   **/
  case class Reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long]) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents a segment of a tensor as a tensor.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `slice` to perform invariant checks.
   **/
  case class Slice(tensor: this.Tensor, sliceRange: Array[(Long, Long)]) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents construction of a tensor from two or more base tensors.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `join` to perform invariant checks.
   **/
  case class Join(tensors: Array[this.Tensor], joiningDimension: Dimension) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor by applying a `MorphFunction` to each of
   * the constituent sub-tensors of a source tensor, and joining the results
   * into a resultant tensor. The interpreter chooses the sequencing and parallelism
   * of the application of the `MorphFunction`.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `morph` to perform invariant checks.
   **/
  // case class Morph(tensor: this.Tensor, grouping: Array[Dimension], morph_f: this.MorphFunction) extends this.TensorExprOp[this.Tensor]

  /**
   * Each algebra must define its own definition of the MorphFunction type. The Algebra should
   * also include a mechanism to construct or obtain instances. Algebras will also have to
   * provide some specific instances: `_SUM`, `_PRODUCT`
   **/
  // type MorphFunction <: bmaso.tensoralg.abstractions.MorphFunction

  /**
   * Represents the nullary operation.
   **/
  case object Unit extends this.TensorExprOp[Unit]

  type TensorExpr[Eff] = Free[TensorExprOp, Eff]

  def translate(tensor: this.Tensor, offsets: Array[Long]): this.TensorExpr[this.Tensor] =
    //...drop trailing zero offsets for efficiency...
    Free.liftF(Translate(tensor, offsets.reverse.dropWhile(_ == 0).reverse))

  def broadcast(tensor: this.Tensor, baseMagnitude: Array[Long]): this.TensorExpr[this.Tensor] = {
    if(!baseMagnitude.map(_ > 0).fold(true)(_ && _)) throw new IllegalArgumentException("Base magntiude includes illegal values; each dimension must be > 0")

    if(baseMagnitude.map(_ == 1).fold(true)(_ && _))
      Free.pure(tensor)
    else
      Free.liftF(Broadcast(tensor, baseMagnitude))
  }

  // TODO: ensure reshaped elementSize is same as original; ensure reshaped magnitude values are non-negative and non-zero; ensure dimension values are valid dimensions (non-zero, non-negative)
  def reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long]): this.TensorExpr[this.Tensor] = Free.liftF(Reshape(tensor, reshapedMagnitude))

  def slice(tensor: this.Tensor, sliceRange: Array[(Long, Long)]): this.TensorExpr[this.Tensor] = {
    if(!sliceRange.drop(tensor.order).map({ case (f, l) => f == 0 && l == 1}).fold(true)(_ && _)) throw new IllegalArgumentException("Slice range implies higher order than source tensor")
    if(!tensor.magnitude.zip(sliceRange).map({ case (m, (f, l)) => f + l <= m}).fold(true)(_ && _)) throw new IllegalArgumentException("Slice range is outside of source tensor magnitude")
    Free.liftF(Slice(tensor, sliceRange))
  }

  // TODO: ensure source tensors are of same magnitude in all but the joining dimension; if joining dimension mag for all tensors is 1 use StackTensor, else use less efficient JoinTensor
  def join(tensors: Array[this.Tensor], joiningDimension: Dimension): this.TensorExpr[this.Tensor] = Free.liftF(Join(tensors, joiningDimension))

  /**
   * Constructs a tensor by reversing the elements in a specific dimension. The same
   * could be accomplished by slicing unitary width tensors in a specific dimension
   * and re-assembling with `join`. This is a much simpler and more efficient way
   * to do this common operation.
   **/
  def reverse(tensor: this.Tensor, dimension: Dimension): this.TensorExpr[this.Tensor] = ???

  /**
   * Pivots a tensor by exchanging 2 dimensions
   **/
  def pivot(tensor: this.Tensor, dim1: Dimension, dim2: Dimension): this.TensorExpr[this.Tensor] = ???

  /**
   * Construction of a tensor by application of a `morph_f` function on all
   * subtensors projected from `subtensorDimension`. The resultant subtensors
   * aare joined, also in `subtensorDimensions`, to form the resultant tensor.
   * Note that the interpreter will pick the sequencing and parallelism for
   * invoking the morphing function on each of the subtensors.
   **/
  def morph(tensor: this.Tensor, subtensorDimensions: Array[Dimension], morph_f: this.MorphFunction): TensorExpr[this.Tensor] = ???

  def unit(): this.TensorExpr[Unit] = Free.liftF(this.Unit)

  // TBD: The following is saved code to be used to implement split, a function
  // that should divide an original tensor into a grid of slices.
  //
  // val splitMagnitude = {
  //   val splitsMap: Map[Dimension, Long] =
  //     splitDimensions zipWith (splitStepping) toMap
  //   //...calc original magnitudes, with split ones replaced by correct fractional size
  //   val allMags =
  //     for(d <- 0 to tensor.order - 1;
  //         m = tensor.magnitude(d)) yield {
  //       splitsMap.getOrElse(d, m)
  //     }
  //   //...truncate list of magnitudes to remove mags reduced to 1
  //   val lastRemovedDimension = allMags.lastIndexOf(1)
  //   allMags.take(lastRemovedDimension)
  // }
  //
  // val splitSteppingIndexes: List[List[Long]] = {
  //   @tailrec
  //   def rec(mags: List[Long]): List[List[Long]] = mags match {
  //     case Nil => List()
  //     case n :: rest =>
  //       val children = rec(rest)
  //       for(i <- 0 to n;
  //           c <- children) yield { n +: c }
  //   }
  //
  //   val splitDimensionCounts: List[Long] =
  //     for((m, d) <- (tensor.magnutide zipWithIndex)) yield {
  //       (m / splitMagnitude(d))
  //     } toList
  //
  //   rec(splitDimensionCounts)
  // }

  //...generating an Array of SplitTensor objects, each one with the same
  //   magnitude and stepping of the original tensor, and each one with a
  //   unique stepping index
  // splitSteppingIndexes.map((steppingIndex: List[Long]) =>
  //     SplitTensor(tensor, splitMagnitude, splitStepping, steppingIndex.toArray))
  //   .toArray


}
