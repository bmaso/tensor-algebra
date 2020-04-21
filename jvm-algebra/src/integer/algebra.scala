package bmaso.tensoralg.jvm.integer

import cats.free.Free
import bmaso.tensoralg.abstractions.{TensorAlgebra => abstract_TensorAlgebra}

object IntTensorAlgebra extends abstract_TensorAlgebra {
   override type Tensor = IntTensor

   case class TensorFromArray(arr: Array[Int], offset: Int, length: Int) extends this.TensorExprOp[this.Tensor]
   case class CopyTensorElementsToArray(tensor: this.Tensor, arr: Array[Int], offset: Int) extends this.TensorExprOp[this.Tensor]

   def tensorFromArray(arr: Array[Int], offset: Int = 0, len: Int = -1): this.TensorExpr[this.Tensor] = Free.liftF(TensorFromArray(arr, offset, len))
   def copyTensorElementsToArray(tensor: this.Tensor, arr: Array[Int], offset: Int = 0): this.TensorExpr[this.Tensor] = Free.liftF(CopyTensorElementsToArray(tensor, arr, offset))
}
