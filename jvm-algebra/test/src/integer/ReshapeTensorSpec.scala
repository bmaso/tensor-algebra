package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

abstract class ReshapeTensorSpecBase[T: Numeric: ClassTag: ConvertsInt] extends TensorFlatSpecBase[T] {
  "A 3x4x5 ReshapeTensor made from a 10x2x3 tensor" should "have the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[T]((0 to 59).toArray.asTArray, Array(10, 2, 3), 0)
    val reshaped = ReshapeTensor[T](arrayTensor, Array(3, 4, 5))

    reshaped.magnitude should be (Array(3, 4, 5))
    reshaped.order should be (3)
    reshaped.elementSize should be (60)

    reshaped.valueAt(Array(0, 0, 0)).asI should be (0)
    reshaped.valueAt(Array(2, 2, 1)).asI should be (20)
    reshaped.valueAt(Array(0, 1, 4)).asI should be (51)
    reshaped.valueAt(Array(1, 3, 2)).asI should be (34)
    reshaped.valueAt(Array(2, 3, 4)).asI should be (59)
  }

  "Providing additional dimension index 0 values when getting values from a ReshapedTensor" should "yield the same as not providing those values" in {
    val arrayTensor = ArrayTensor[T]((0 to 59).toArray.asTArray, Array(10, 2, 3), 0)
    val reshaped = ReshapeTensor[T](arrayTensor, Array(3, 4, 5))

    reshaped.valueAt(Array(0, 0, 0)) should be (reshaped.valueAt(Array(0, 0, 0, 0)))
    reshaped.valueAt(Array(2, 2, 1)) should be (reshaped.valueAt(Array(2, 2, 1, 0, 0)))
    reshaped.valueAt(Array(0, 1, 4)) should be (reshaped.valueAt(Array(0, 1, 4, 0, 0, 0)))
    reshaped.valueAt(Array(1, 3, 2)) should be (reshaped.valueAt(Array(1, 3, 2, 0)))
    reshaped.valueAt(Array(2, 3, 4)) should be (reshaped.valueAt(Array(2, 3, 4, 0, 0, 0, 0)))
  }
}

class ByteReshapeTensorSpecBase extends ReshapeTensorSpecBase[Byte]
class ShortReshapeTensorSpecBase extends ReshapeTensorSpecBase[Short]
class IntReshapeTensorSpecBase extends ReshapeTensorSpecBase[Int]
class LongReshapeTensorSpecBase extends ReshapeTensorSpecBase[Long]
class FloatReshapeTensorSpecBase extends ReshapeTensorSpecBase[Float]
class DoubleReshapeTensorSpecBase extends ReshapeTensorSpecBase[Double]
