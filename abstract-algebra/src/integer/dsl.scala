package bmaso.tensoralg.abstractions.integer

import cats.free.Free
import bmaso.tensoralg.abstractions.{Tensor, IntElT, Dimension, NaturalDimensionSequencing, UnitaryStepping, MaximalGrouping}

object dsl {
  def tensorFromArray(arr: Array[Int], offset: Int, len: Int): IntTensorExpr[Tensor[IntElT]] =
      Free.liftF(TensorFromArray(arr, offset, len))
  def copyTensorElementsToArray(tensor: Tensor[IntElT], arr: Array[Int], offset: Int): IntTensorExpr[Tensor[IntElT]] =
      Free.liftF(CopyTensorElementsToArray(tensor, arr, offset))
  def translate(tensor: Tensor[IntElT], dimensions: Array[Dimension], offsets: Array[Long]): IntTensorExpr[Tensor[IntElT]] =
      Free.liftF(Translate(tensor, dimensions, offsets))
  def broadcast(tensor: Tensor[IntElT], baseMagnitude: Array[Long]): IntTensorExpr[Tensor[IntElT]] =
      Free.liftF(Broadcast(tensor, baseMagnitude))
  def reshape(tensor: Tensor[IntElT], reshapedMagnitude: Array[Long], sourceDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing, targetDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing): IntTensorExpr[Tensor[IntElT]] =
      Free.liftF(Reshape(tensor, reshapedMagnitude, sourceDimensionSequencing, targetDimensionSequencing))
  def split(tensor: Tensor[IntElT], splitDimensions: Array[Dimension], splitStepping: Array[Long] = UnitaryStepping): IntTensorExpr[Array[Tensor[IntElT]]] =
      Free.liftF(Split(tensor, splitDimensions, splitStepping))
  def join(tensors: Array[Tensor[IntElT]], joiningDimensions: Array[Dimension], joinGrouping: Array[Long] = MaximalGrouping): IntTensorExpr[Tensor[IntElT]] =
      Free.liftF(Join(tensors, joiningDimensions, joinGrouping))
}
