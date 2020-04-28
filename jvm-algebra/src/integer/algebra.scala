package bmaso.tensoralg.jvm.integer

import cats.free.Free
import bmaso.tensoralg.abstractions.{TensorAlgebra => abstract_TensorAlgebra, Dimension}

trait IntTensorAlgebra extends abstract_TensorAlgebra {
   override type Tensor = IntTensor
   override type MorphFunction = IntMorphFunction

   case class TensorFromArray(arr: Array[Int], magnitude: Array[Long], offset: Int) extends this.TensorExprOp[this.Tensor]
   case class CopyTensorElementsToArray(tensor: this.Tensor, arr: Array[Int], offset: Int) extends this.TensorExprOp[this.Tensor]

   def tensorFromArray(arr: Array[Int], magnitude: Array[Long], offset: Int = 0): this.TensorExpr[this.Tensor] = Free.liftF(TensorFromArray(arr, magnitude, offset))
   def copyTensorElementsToArray(tensor: this.Tensor, arr: Array[Int], offset: Int = 0): this.TensorExpr[this.Tensor] = Free.liftF(CopyTensorElementsToArray(tensor, arr, offset))

   /**
    * Simple DSL for constructing a representation of a morph using a Scala function
    * to transform an input tensor. The transformed tensor is memoized to an
    * array-backed tensor.
    *
    * Example use of the DSL to define a morph:
    *
    * Assuming `mySrcTensor` is an ***N***x***M***x9x6 tensor, this operation describes
    * a transformation to an ***N***x***M***x3 tensor memoized by backing Array
    * `someArray` starting at array offset 0.
    *
    * ```
    * morph(mySrcTensor)
    *     slicing(Array(_X, _Y))
    *     withTransformedSubtensorMagnitude(Array(1, 1, 3))
    *     toArray(someArray) { subtensor =>
    *   //...transform subtensor 1x1x9x6 => 1x1x3
    * }
    * ```
    **/
   def morph(tensor: this.Tensor) = new {
     def slicing(subtensorBaseDimensions: Array[Dimension]) = new {
       def withTransformedSubtensorMagnitude(magnitude: Array[Long]) = new {
         def toArray(arr: Array[Int], offset: Int = 0)(f: (IntTensor) => IntTensor): this.TensorExpr[this.Tensor] = {
           // TODO: validate the magnitudes and dimensions against each other

           val morph_f = new IntMorphToArrayFunction {
             override val backingArray = arr
             override val backingArrayOffset = offset
             override lazy val resultMagnitude: Array[Long] = ???

             override val inputTensorMagnitude: Array[Long] = {
               //...original tensor magnitudes for all subtensorBaseDimensions, and unitary
               //   magnitude for all other dimensions...
               val ret: Array[Long] = Array.fill[Long](subtensorBaseDimensions.max + 1)(1)
               for(d <- subtensorBaseDimensions) ret(d) = tensor.magnitudeIn(d)
               ret
             }
             override val outputTensorMagnitude: Array[Long] = magnitude
             override def apply(t: this.Tensor): this.Tensor = f(t)
           }
           _morph(tensor, subtensorBaseDimensions, transformedSubtensorMagnitude, morph_f)
         }
       }
     }
   }
}

object IntTensorAlgebra extends IntTensorAlgebra
