package bmaso.tensoralg.jvm.integer

import cats.free.Free
import bmaso.tensoralg.abstractions.{TensorAlgebra => abstract_TensorAlgebra, Dimension}

trait TensorAlgebra[T] extends abstract_TensorAlgebra {
   override type Tensor = JVMTensor[T]

   override type MapFunction = JVMMapFunction
   case class JVMMapFunction(f: (T) => T)

   override type ReduceFunction = JVMReduceFunction
   case class JVMReduceFunction(f: (this.Tensor) => T)

   case class TensorFromArray(arr: Array[T], magnitude: Array[Long], offset: Int) extends this.TensorExprOp[this.Tensor]
   case class CopyTensorElementsToArray(tensor: this.Tensor, arr: Array[T], offset: Int) extends this.TensorExprOp[this.Tensor]

   def tensorFromArray(arr: Array[T], magnitude: Array[Long], offset: Int = 0): this.TensorExpr[this.Tensor] =
     Free.liftF(TensorFromArray(arr, magnitude, offset))
   def copyTensorElementsToArray(tensor: this.Tensor, arr: Array[T], offset: Int = 0): this.TensorExpr[this.Tensor] =
     Free.liftF(CopyTensorElementsToArray(tensor, arr, offset))

   def map(tensor: this.Tensor)(f: (T) => T): this.TensorExpr[this.Tensor] = {
     val map_f: this.MapFunction = JVMMapFunction(f)
     super.map(tensor, map_f)
   }

   def reduce(tensor: this.Tensor, reduceOrders: Int)(f: (this.Tensor) => T): this.TensorExpr[this.Tensor] = {
     val reduce_f: this.ReduceFunction = JVMReduceFunction(f)
     super.reduce(tensor, reduceOrders, reduce_f)
   }
}

object ByteTensorAlgebra extends TensorAlgebra[Byte] {
  override lazy val SUM = JVMReduceFunction(t =>
    (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).sum)
  override lazy val PRODUCT = JVMReduceFunction(t =>
    (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).foldLeft(1.toByte)({ case (acc, e) => (acc * e).toByte }))
}

object ShortTensorAlgebra extends TensorAlgebra[Short] {
  override lazy val SUM = JVMReduceFunction(t =>
    (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).sum)
  override lazy val PRODUCT = JVMReduceFunction(t =>
    (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).foldLeft(1.toShort)({ case (acc, e) => (acc * e).toShort }))
}

object IntTensorAlgebra extends TensorAlgebra[Int] {
  override lazy val SUM = JVMReduceFunction(t =>
    (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).sum)
  override lazy val PRODUCT = JVMReduceFunction(t =>
    (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).foldLeft(1)(_ * _))
}

object LongTensorAlgebra extends TensorAlgebra[Long] {
  override lazy val SUM = JVMReduceFunction(t =>
    (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).sum)
  override lazy val PRODUCT = JVMReduceFunction(t =>
    (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).foldLeft(1L)(_ * _))
}
