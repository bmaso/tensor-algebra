package bmaso.tensoralg.jvm.integer

import cats.{~>, Id}
import cats.implicits._

import IntTensorAlgebra._

object IdInterpreter extends (TensorExprOp ~> Id) {

  def eval[A](expr: TensorExpr[A]): Id[A] = expr.foldMap(this)

  override def apply[A](expr: TensorExprOp[A]): Id[A] = expr match {
    case TensorFromArray(array, magnitude, offset, length) =>
      IntArrayTensor(array, magnitude, offset, length)

    case CopyTensorElementsToArray(tensor, targetArray, targetOffset) => tensor match {
      case IntArrayTensor(array, magnitude, offset, length) =>
        Array.copy(array, offset, targetArray, targetOffset, length)
        tensor

      case t: IntTensor => ???
    }
  }
}
