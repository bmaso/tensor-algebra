package bmaso.tensoralg.jvm.integer

import cats.free.Free
import bmaso.tensoralg.abstractions.{TensorAlgebra => abstract_TensorAlgebra, Dimension}

trait IntTensorAlgebra extends abstract_TensorAlgebra {
   override type Tensor = IntTensor

   override type MapFunction = IntMapFunction
   case class IntMapFunction(f: (Int) => Int)

   override type ReduceFunction = IntReduceFunction
   case class IntReduceFunction(f: (this.Tensor) => Int)

   override lazy val SUM = IntReduceFunction(t =>
     (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).sum)
   override lazy val PRODUCT = IntReduceFunction(t =>
     (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).foldLeft(1)(_ * _))

   case class TensorFromArray(arr: Array[Int], magnitude: Array[Long], offset: Int) extends this.TensorExprOp[this.Tensor]
   case class CopyTensorElementsToArray(tensor: this.Tensor, arr: Array[Int], offset: Int) extends this.TensorExprOp[this.Tensor]

   def tensorFromArray(arr: Array[Int], magnitude: Array[Long], offset: Int = 0): this.TensorExpr[this.Tensor] = Free.liftF(TensorFromArray(arr, magnitude, offset))
   def copyTensorElementsToArray(tensor: this.Tensor, arr: Array[Int], offset: Int = 0): this.TensorExpr[this.Tensor] = Free.liftF(CopyTensorElementsToArray(tensor, arr, offset))

   def map(tensor: this.Tensor)(f: (Int) => Int): this.TensorExpr[this.Tensor] = {
     val map_f: this.MapFunction = IntMapFunction(f)
     super.map(tensor, map_f)
   }

   def reduce(tensor: this.Tensor, reduceOrders: Int)(f: (this.Tensor) => Int): this.TensorExpr[this.Tensor] = {
     val reduce_f: this.ReduceFunction = IntReduceFunction(f)
     super.reduce(tensor, reduceOrders, reduce_f)
   }
}

object IntTensorAlgebra extends IntTensorAlgebra
