package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.{~>, Id}
import cats.implicits._

import bmaso.tensoralg.abstractions.Dimension
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
      //...TODO: ensure slice range is not outside source tensor's bounds...
      SliceTensor(tensor, sliceRange)

    case Join(tensors: Array[tensor], joiningDimension: Dimension) =>
      ???
      //...TODO: ensure source tensors are of same size in all but the joining dimension...

      //...use a StackTensor to represent joining if the source tensors are unitary
      //   in the join dimension -- this is much more efficient in space and
      //   access time...


      //...when source tensors are not unitary in join dimension, then use JoinTensor,
      //   which has the added capability to stack tensors of uneven size in the
      //   join dimension...

    case IntTensorAlgebra.Unit => ()
  }
}
