package bmaso.tensoralg.jvm.integer

import cats.{~>, Id}
import cats.implicits._

import IntTensorAlgebra._

object IdInterpreter extends (TensorExprOp ~> Id) {

  def eval[A](expr: TensorExpr[A]): Id[A] = expr.foldMap(this)

  override def apply[A](expr: TensorExprOp[A]): Id[A] = expr match {
    case TensorFromArray(array, magnitude, offset) =>
      IntArrayTensor(array, magnitude, offset)

    case CopyTensorElementsToArray(tensor, targetArray, targetOffset) => tensor match {
      case IntArrayTensor(array, magnitude, offset) =>
        Array.copy(array, offset, targetArray, targetOffset, tensor.elementSize.toInt)
        tensor

      case t: IntTensor =>
        for(idx <- 0 to t.elementSize.toInt - 1) {
          targetArray(idx + targetOffset) = t.valueAt1D(idx)
        }
        t
    }

    case Translate(tensor: IntTensor, offsets: Array[Long]) =>
      TranslateTensor(tensor: IntTensor, offsets: Array[Long])

    case Broadcast(tensor: IntTensor, baseMagnitude: Array[Long]) =>
      BroadcastTensor(tensor, baseMagnitude)

    case IntTensorAlgebra.Unit => ()
  }
}
