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

  //...tensor shape and element index gymnastic operators...
  case class Translate(tensor: this.Tensor, offsets: Array[Long]) extends this.TensorExprOp[this.Tensor]
  case class Broadcast(tensor: this.Tensor, baseMagnitude: Array[Long]) extends this.TensorExprOp[this.Tensor]
  case class Reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long], sourceDimensionSequencing: Array[Dimension], targetDimensionSequencing: Array[Dimension]) extends this.TensorExprOp[this.Tensor]
  case class Slice(tensor: this.Tensor, sliceRange: Array[(Long, Long)]) extends this.TensorExprOp[this.Tensor]
  case class Join(tensors: Array[this.Tensor], joiningDimension: Dimension) extends this.TensorExprOp[this.Tensor]

  //...tensor control and flow operators...
  case object Unit extends this.TensorExprOp[Unit]

  type TensorExpr[Eff] = Free[TensorExprOp, Eff]

  def translate(tensor: this.Tensor, offsets: Array[Long]): this.TensorExpr[this.Tensor] = Free.liftF(Translate(tensor, offsets))
  def broadcast(tensor: this.Tensor, baseMagnitude: Array[Long]): this.TensorExpr[this.Tensor] = Free.liftF(Broadcast(tensor, baseMagnitude))
  def reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long], sourceDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing, targetDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing): this.TensorExpr[this.Tensor] = Free.liftF(Reshape(tensor, reshapedMagnitude, sourceDimensionSequencing, targetDimensionSequencing))
  def slice(tensor: this.Tensor, sliceRange: Array[(Long, Long)]): this.TensorExpr[this.Tensor] = Free.liftF(Slice(tensor, sliceRange))
  def join(tensors: Array[this.Tensor], joiningDimension: Dimension): this.TensorExpr[this.Tensor] = Free.liftF(Join(tensors, joiningDimension))
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
