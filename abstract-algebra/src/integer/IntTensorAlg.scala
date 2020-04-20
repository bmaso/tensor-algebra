package bmaso.tensoralg.abstractions.integer

import bmaso.tensoralg.abstractions.{Dimension, NaturalDimensionSequencing, UnitaryStepping, MaximalGrouping, Tensor, IntElT}
/**
 * The tensor algebraic expressions which are common to all evaluators. Using the
 * *Free monad* pattern: allows me to compose evaluator-specific algebras
 * in order to specialize ***reduce*** operation definitions for different
 * evaluators.
 **/
sealed trait IntTensorExprAlg[T]
case class TensorFromArray(arr: Array[Int], offset: Int, len: Int) extends IntTensorExprAlg[Tensor[IntElT]]
case class CopyTensorElementsToArray(tensor: Tensor[IntElT], arr: Array[Int], offset: Int) extends IntTensorExprAlg[Tensor[IntElT]]
case class Translate(tensor: Tensor[IntElT], dimensions: Array[Dimension], offsets: Array[Long]) extends IntTensorExprAlg[Tensor[IntElT]]
case class Broadcast(tensor: Tensor[IntElT], baseMagnitude: Array[Long]) extends IntTensorExprAlg[Tensor[IntElT]]
case class Reshape(tensor: Tensor[IntElT], reshapedMagnitude: Array[Long], sourceDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing, targetDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing) extends IntTensorExprAlg[Tensor[IntElT]]
case class Split(tensor: Tensor[IntElT], splitDimensions: Array[Dimension], splitStepping: Array[Long] = UnitaryStepping) extends IntTensorExprAlg[Array[Tensor[IntElT]]]
case class Join(tensors: Array[Tensor[IntElT]], joiningDimensions: Array[Dimension], joinGrouping: Array[Long] = MaximalGrouping) extends IntTensorExprAlg[Tensor[IntElT]]
