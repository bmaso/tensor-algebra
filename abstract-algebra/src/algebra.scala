package bmaso.tensoralg.abstractions

import cats.free.Free

/**
 * Each evaluator defines its own concrete tensor type, and defines its operations
 * in terms of this tensor type.
 **/
trait TensorAlgebra {
  type Tensor <: bmaso.tensoralg.abstractions.Tensor
  // type AggregatingTensor <: this.Tensor with Aggregating[this.Tensor]
  // trait AggregatingTensorBuilder {
  //   def freshTensor(magnitude: Array[Long]): TensorAlgebra.this.AggregatingTensor
  // }

  /**
   * Ambient object able to create a "fresh" aggregating tensor. The aggregating
   * tensor should have a "zero" value, meaning that after it aggregates a tensor
   * ***A***, then the element values for the resultant aggregate should be identical
   * to the element values of ***A***.
   **/
  // val aggregatingTensorBuilder: AggregatingTensorBuilder

  type MapFunction
  // type ReduceFunction
  // type TransformFunction

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
  case class Join(tensors: List[this.Tensor], joiningDimension: Dimension) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor by 1:1 mapping of element values.
   *
   * @returns The resulting tensor has the exact same magnitude as the input
   *     tensor.
   **/
  case class Map(tensor: this.Tensor, map_f: this.MapFunction) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor by reducing a source tensor in zero
   * or more dimensions using a reducing function.
   *
   * @returns The resulting tensor is of order `tensor.order - reducedOrders`. The
   *     magnitude or the resulting tensor will be equal to `tensor.magnitude.take(tensor.order - reducedOrders)`
   * @param reduce_f The reducing function must accept tensors of magnitude
   *     `tensor.magnitude.drop(tensor.order - reducedOrders)`
   **/
  // case class Reduce(tensor: this.Tensor, reducedOrders: Int, reduce_f: this.ReduceFunction) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor by map-reduce. Conceptually, individual
   * slices of the source tensor are transformed via a transforming function,
   * and the results are aggregated together into an aggregating tensor.
   *
   * @returns The resulting tensor is of magnitude `producedMagnitude`
   *
   * @param transform_f The transforming function must accept tensors of magnitude
   *     `tensor.magnitude.drop(tensor.order - reducedOrders)`, and produce tensors
   *     of magnitude `producedMagnitude`
   * @param builder The tensor builder must generate "zero-valued" `AggregateTensor`
   *     instances of magnitude `producedMagnitude`
   **/
  // case class Aggregate(tensor: this.Tensor, reducedOrders: Int, producedMagnitude: Array[Long], transform_f: this.TransformFunction) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor from another by reversing index values
   * along one dimension.
   **/
  case class Reverse(tensor: this.Tensor, dimension: Dimension) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of one tensor from another by exchanging the
   * indexes in exactly 2 dimensions.
   **/
  case class Pivot(tensor: this.Tensor, dim1: Dimension, dim2: Dimension) extends this.TensorExprOp[this.Tensor]

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

  /**
   * Construct a tensor with the same magnitude as the original, but with values
   * tanslated in zero or more dimensions. Element values translated to indexes
   * outside original magnitude will be truncated. Resultant tensor will be "backfilled"
   * with zero values for elements that don't have a corresponding value in original
   * tensor.
   **/
  def translate(tensor: this.Tensor, offsets: Array[Long]): this.TensorExpr[this.Tensor] =
    //...drop trailing zero offsets for efficiency...
    Free.liftF(Translate(tensor, offsets.reverse.dropWhile(_ == 0).reverse))

  /**
   * Construct a tensor of higher dimensionality that the original by duplication
   * in zero or more dimensions.
   *
   * Note that the dual operation would be to remove lower dimension(s) of magnitude
   * 1. For example, turning a 1x1x3x3 tensor into a 3x3 tensor. This can be
   * accomplished using the `pivot` operation.
   **/
  def broadcast(tensor: this.Tensor, baseMagnitude: Array[Long]): this.TensorExpr[this.Tensor] = {
    if(!baseMagnitude.map(_ > 0).fold(true)(_ && _)) throw new IllegalArgumentException("Base magntiude includes illegal values; each dimension must be > 0")

    if(baseMagnitude.map(_ == 1).fold(true)(_ && _))
      Free.pure(tensor)
    else
      Free.liftF(Broadcast(tensor, baseMagnitude))
  }

  def reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long]): this.TensorExpr[this.Tensor] = {
    if(!reshapedMagnitude.map(_ >= 1).fold(true)(_ && _)) throw new IllegalArgumentException("Reshaped magnitude includes illegal values; all arities must be >= 1")
    if(tensor.elementSize != reshapedMagnitude.fold(1L)(_ * _)) throw new IllegalArgumentException("Reshaped tensor elementSize must equal source elementSize")

    Free.liftF(Reshape(tensor, reshapedMagnitude))
  }

  def slice(tensor: this.Tensor, sliceRange: Array[(Long, Long)]): this.TensorExpr[this.Tensor] = {
    if(!sliceRange.drop(tensor.order).map({ case (f, l) => f == 0 && l == 1}).fold(true)(_ && _)) throw new IllegalArgumentException("Slice range implies higher order than source tensor")
    if(!tensor.magnitude.zip(sliceRange).map({ case (m, (f, l)) => f + l <= m}).fold(true)(_ && _)) throw new IllegalArgumentException("Slice range is outside of source tensor magnitude")
    Free.liftF(Slice(tensor, sliceRange))
  }

  def join(joiningDimension: Dimension, t1: this.Tensor, t2: this.Tensor*): this.TensorExpr[this.Tensor] = {
    val tensors: List[this.Tensor] = List(t1) :++ t2.toList
    Free.liftF(Join(tensors, joiningDimension))
  }

  /**
   * Constructs a tensor by reversing the elements in a specific dimension.
   *
   * "Rotating" a tensor about one or more dimensions can be accomplished by
   * applying different sequences of `reverse` and `pivot`.
   *
   * Note that the same could be accomplished by slicing unitary width tensors in
   * a specific dimension and re-assembling with `join` in opposite order. Defining
   * `reverse` as a basic operation makes this common operation much easier to
   * implement efficiently by initerpreters.
   **/
  def reverse(tensor: this.Tensor, dimension: Dimension): this.TensorExpr[this.Tensor] = {
    if(dimension < 0) throw new IllegalArgumentException("Dimension value is not legal; dimension numbers must be >= 0")
    Free.liftF(Reverse(tensor, dimension))
  }

  /**
   * Pivots a tensor by exchanging 2 dimensions.
   *
   * "Rotating" a tensor about one or more dimensions can be accomplished by
   * applying different sequences of `reverse` and `pivot`.
   **/
  def pivot(tensor: this.Tensor, dim1: Dimension, dim2: Dimension): this.TensorExpr[this.Tensor] = {
    if(dim1 < 0 || dim2 < 0) throw new IllegalArgumentException("One or both dimension value is illegal; dimension numbers must be >= 0")
    Free.liftF(Pivot(tensor, dim1, dim2))
  }

  /**
   * Constructs a new tensor whose elements are a 1:1 mapping of the source
   * tensor.
   * @returns An expression that, when evaluated, yields a tensor with the exact
   *     same magnitude as `tensor.magnitude`
   **/
  def map(tensor: this.Tensor, map_f: this.MapFunction): this.TensorExpr[this.Tensor] =
    Free.liftF(Map(tensor, map_f))

  /**
   * Constructs a new tensor whose elements are a reduction of slices that comprise
   * the source tensor. The slices are defined by iteating over unitary width
   * slices in the first `reduceOrders` dimensions.
   *
   * @returns An expression that, once evaluated, yields a tensor whose magnitude
   *     is `tensor.magnitude.drop(reduceOrders)`
   * @param reduce_f A reduction function that receives slices from the original
   *     source tensor of magnitude `tensor.magnitude.take(reduceOrders)`, and which
   *     yields a unitary value.
   **/
  // def reduce(tensor: this.Tensor, reduceOrders: Int, reduce_f: this.ReduceFunction): this.TensorExpr[this.Tensor] = ???

  /**
   * Constructs a new tensor built by aggregating transformed slices of the original
   * source tensor.
   *
   * @returns The returned expression, whenn evaluated, yields a tensor of magnitude
   *     `producedMagnitude`
   * @param transform_f A tranform function that receives a tensor of magnitude
   *     `tensor.magnitude.take(reduceOrders)`, and which yields a tensor of
   *     magnitude `producedMagnitude`.
   **/
  // def aggregate(tensor: this.Tensor, reduceOrders: Int, produceMagnitude: Array[Long], transform_f: this.TransformFunction): this.TensorExpr[this.Tensor] = ???

  /**
   * Provides a way for an expression to result in "nothing", relying completely
   * on interpreter-specific operation side-effects.
   **/
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
