package bmaso.tensoralg.abstractions

import cats.free.Free

object dsl {
  /*
   * Coding note: Lots of type-fu going on here. Basically we need to explicitly tell the
   * compiler what conclusions and type bindings we want because it's too complicated
   * for the compiler to work out on its own.
   */

  def tensorFromArray[N, ElT <: TensorElT](arr: Array[N], offset: Int, len: Int)(implicit ev: ScalaElTMatches[N, ElT]): TensorExpr[ElT, Tensor[ElT]] =
      Free.liftF(TensorFromArray(arr, offset, len): ({type λ[A] = TensorExprAlg[ElT, A] })#λ[Tensor[ElT]])
  def copyTensorElementsToArray[N, ElT <: TensorElT](tensor: Tensor[ElT], arr: Array[N], offset: Int)(implicit ev: ScalaElTMatches[N, ElT]): TensorExpr[ElT, Either[ComputationError, Tensor[ElT]]] =
      Free.liftF(CopyTensorElementsToArray(tensor, arr, offset): ({type λ[A] = TensorExprAlg[ElT, A] })#λ[Either[ComputationError, Tensor[ElT]]])
  def translate[ElT <: TensorElT](tensor: Tensor[ElT], dimensions: Array[Dimension], offsets: Array[Long]): TensorExpr[ElT, Tensor[ElT]] =
      Free.liftF(Translate(tensor, dimensions, offsets): ({type λ[A] = TensorExprAlg[ElT, A] })#λ[Tensor[ElT]])
  def broadcast[ElT <: TensorElT](tensor: Tensor[ElT], baseMagnitude: Array[Long]): TensorExpr[ElT, Tensor[ElT]] =
      Free.liftF(Broadcast(tensor, baseMagnitude): ({type λ[A] = TensorExprAlg[ElT, A] })#λ[Tensor[ElT]])
  def reshape[ElT <: TensorElT](tensor: Tensor[ElT], reshapedMagnitude: Array[Long], sourceDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing, targetDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing): TensorExpr[ElT, Tensor[ElT]] =
      Free.liftF(Reshape(tensor, reshapedMagnitude, sourceDimensionSequencing, targetDimensionSequencing): ({type λ[A] = TensorExprAlg[ElT, A] })#λ[Tensor[ElT]])
  def split[ElT <: TensorElT](tensor: Tensor[ElT], splitDimensions: Array[Dimension], splitStepping: Array[Long] = UnitaryStepping): TensorExpr[ElT, Array[Tensor[ElT]]] =
      Free.liftF(Split(tensor, splitDimensions, splitStepping): ({type λ[A] = TensorExprAlg[ElT, A] })#λ[Array[Tensor[ElT]]])
  def join[ElT <: TensorElT](tensors: Array[Tensor[ElT]], joiningDimensions: Array[Dimension], joinGrouping: Array[Long] = MaximalGrouping): TensorExpr[ElT, Tensor[ElT]] =
      Free.liftF(Join(tensors, joiningDimensions, joinGrouping): ({type λ[A] = TensorExprAlg[ElT, A] })#λ[Tensor[ElT]])
}
