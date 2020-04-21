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
  case class Translate(tensor: this.Tensor, dimensions: Array[Dimension], offsets: Array[Long]) extends this.TensorExprOp[this.Tensor]
  case class Broadcast(tensor: this.Tensor, baseMagnitude: Array[Long]) extends this.TensorExprOp[this.Tensor]
  case class Reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long], sourceDimensionSequencing: Array[Dimension], targetDimensionSequencing: Array[Dimension]) extends this.TensorExprOp[this.Tensor]
  case class Split(tensor: this.Tensor, splitDimensions: Array[Dimension], splitStepping: Array[Long]) extends this.TensorExprOp[Array[this.Tensor]]
  case class Join(tensors: Array[this.Tensor], joiningDimensions: Array[Dimension], joinGrouping: Array[Long]) extends this.TensorExprOp[this.Tensor]

  type TensorExpr[Eff] = Free[TensorExprOp, Eff]

  def translate(tensor: this.Tensor, dimensions: Array[Dimension], offsets: Array[Long]): this.TensorExpr[this.Tensor] = Free.liftF(Translate(tensor, dimensions, offsets))
  def broadcast(tensor: this.Tensor, baseMagnitude: Array[Long]): this.TensorExpr[this.Tensor] = Free.liftF(Broadcast(tensor, baseMagnitude))
  def reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long], sourceDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing, targetDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing): this.TensorExpr[this.Tensor] = Free.liftF(Reshape(tensor, reshapedMagnitude, sourceDimensionSequencing, targetDimensionSequencing))
  def split(tensor: this.Tensor, splitDimensions: Array[Dimension], splitStepping: Array[Long] = UnitaryStepping): this.TensorExpr[Array[this.Tensor]] = Free.liftF(Split(tensor, splitDimensions, splitStepping))
  def join(tensors: Array[this.Tensor], joiningDimensions: Array[Dimension], joinGrouping: Array[Long] = MaximalGrouping): this.TensorExpr[this.Tensor] = Free.liftF(Join(tensors, joiningDimensions, joinGrouping))
}
