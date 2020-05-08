package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._


abstract class SliceTensorSpecBase[T: Numeric: ClassTag: ConvertsInt] extends TensorFlatSpecBase[T] {
  "A 2x2x2 slice of a 4x4x4 tensor" should "have the right magnitude, order, elementSize, and element values" in {
    val arrayTensor = ArrayTensor[T]((0 to 63).toArray.asTArray, Array(4, 4, 4), 0)
    val slice = SliceTensor[T](arrayTensor, Array((1, 2), (1, 2), (1, 2)))

    slice.magnitude should be (Array(2, 2, 2))
    slice.order should be (3)
    slice.elementSize should be (8)

    slice.valueAt(Array(0, 0, 0)).asI should be (21)
    slice.valueAt(Array(1, 0, 1)).asI should be (38)
    slice.valueAt(Array(0, 1, 0)).asI should be (25)
  }

  "A 2x2x1 slice of a 4x4x4 tensor" should "have the right magnitude, order, elementSize, and element values" in {
    val arrayTensor = ArrayTensor[T]((0 to 63).toArray.asTArray, Array(4, 4, 4), 0)
    val slice = SliceTensor[T](arrayTensor, Array((1, 2), (1, 2), (1, 1)))

    slice.magnitude should be (Array(2, 2))
    slice.order should be (2)
    slice.elementSize should be (4)

    slice.valueAt(Array(0, 0)).asI should be (21)
    slice.valueAt(Array(1, 0)).asI should be (22)
    slice.valueAt(Array(0, 1)).asI should be (25)
    slice.valueAt(Array(1, 1)).asI should be (26)
  }

  "A 2x1x2 slice of a 4x4x4 tensor" should "have the right magnitude, order, elementSize, and element values" in {
    val arrayTensor = ArrayTensor[T]((0 to 63).toArray.asTArray, Array(4, 4, 4), 0)
    val slice = SliceTensor[T](arrayTensor, Array((1, 2), (1, 1), (1, 2)))

    slice.magnitude should be (Array(2, 1, 2))
    slice.order should be (3)
    slice.elementSize should be (4)

    slice.valueAt(Array(0, 0, 0)).asI should be (21)
    slice.valueAt(Array(1, 0, 0)).asI should be (22)
    slice.valueAt(Array(0, 0, 1)).asI should be (37)
    slice.valueAt(Array(1, 0, 1)).asI should be (38)
  }
}

class ByteSliceTensorSpec extends SliceTensorSpecBase[Byte]
class ShortSliceTensorSpec extends SliceTensorSpecBase[Short]
class IntSliceTensorSpec extends SliceTensorSpecBase[Int]
class LongSliceTensorSpec extends SliceTensorSpecBase[Long]
class FloatSliceTensorSpec extends SliceTensorSpecBase[Float]
class DoubleSliceTensorSpec extends SliceTensorSpecBase[Double]
