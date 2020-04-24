package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.{~>, Id}
import cats.implicits._

import IntTensorAlgebra._

/**
 * Note: split operation evaluator assumes split step evenly divides
 * magnitude in the same dimension of original tensor. Code constructing the
 * Split operation needs to assert this assumption.
 **/
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

    case Slice(tensor: IntTensor, sliceRange: Array[(Long, Long)]) =>
      SliceTensor(tensor, sliceRange)
      
    case IntTensorAlgebra.Unit => ()
  }
}
