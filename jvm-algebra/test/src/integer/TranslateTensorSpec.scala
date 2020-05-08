package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

class TranslateTensorSpecBase[T: Numeric: ClassTag: ConvertsInt] extends TensorFlatSpecBase[T] {
  "A translated 3x3 tensor" should "have the expected magnitude, order, elementSize and element values" in {
    val inputTensor = ArrayTensor[T]((1 to 9).toArray.asTArray, Array(3, 3), 0)
    val translation = TranslateTensor[T](inputTensor, Array(1, 1), 0.asT)

    translation.magnitude should be (Array(3, 3))
    translation.order should be (2)
    translation.elementSize should be (9)

    translation.valueAt(Array(0, 0)).asI should be (0)
    translation.valueAt(Array(0, 1)).asI should be (0)
    translation.valueAt(Array(0, 2)).asI should be (0)
    translation.valueAt(Array(1, 0)).asI should be (0)
    translation.valueAt(Array(1, 1)).asI should be (1)
    translation.valueAt(Array(1, 2)).asI should be (4)
    translation.valueAt(Array(2, 0)).asI should be (0)
    translation.valueAt(Array(2, 1)).asI should be (2)
    translation.valueAt(Array(2, 2)).asI should be (5)
  }

  "A translated 3x3 tensor with a negative offset" should "have the expected magnitude, order, elementSize and element values" in {
    val inputTensor = ArrayTensor[T]((1 to 9).toArray.asTArray, Array(3, 3), 0)
    val translation = TranslateTensor[T](inputTensor, Array(-1, 1), 0.asT)

    translation.magnitude should be (Array(3, 3))
    translation.order should be (2)
    translation.elementSize should be (9)

    translation.valueAt(Array(0, 0)).asI should be (0)
    translation.valueAt(Array(0, 1)).asI should be (2)
    translation.valueAt(Array(0, 2)).asI should be (5)
    translation.valueAt(Array(1, 0)).asI should be (0)
    translation.valueAt(Array(1, 1)).asI should be (3)
    translation.valueAt(Array(1, 2)).asI should be (6)
    translation.valueAt(Array(2, 0)).asI should be (0)
    translation.valueAt(Array(2, 1)).asI should be (0)
    translation.valueAt(Array(2, 2)).asI should be (0)
  }

  "A translated 3x3 tensor with offset only the _X dimension" should "have the expected magnitude, order, elementSize and element values" in {
    val inputTensor = ArrayTensor[T]((1 to 9).toArray.asTArray, Array(3, 3), 0)
    val translation = TranslateTensor[T](inputTensor, Array(1), 0.asT)

    translation.magnitude should be (Array(3, 3))
    translation.order should be (2)
    translation.elementSize should be (9)

    translation.valueAt(Array(0, 0)).asI should be (0)
    translation.valueAt(Array(0, 1)).asI should be (0)
    translation.valueAt(Array(0, 2)).asI should be (0)
    translation.valueAt(Array(1, 0)).asI should be (1)
    translation.valueAt(Array(1, 1)).asI should be (4)
    translation.valueAt(Array(1, 2)).asI should be (7)
    translation.valueAt(Array(2, 0)).asI should be (2)
    translation.valueAt(Array(2, 1)).asI should be (5)
    translation.valueAt(Array(2, 2)).asI should be (8)
  }

  it should "ignore additional zero-valued index elements" in {
    val inputTensor = ArrayTensor[T]((1 to 9).toArray.asTArray, Array(3, 3), 0)
    val translation = TranslateTensor[T](inputTensor, Array(1, 1), 0.asT)

    translation.valueAt(Array(0, 0, 0, 0)).asI should be (0)
    translation.valueAt(Array(0, 1, 0, 0)).asI should be (0)
    translation.valueAt(Array(0, 2, 0, 0)).asI should be (0)
    translation.valueAt(Array(1, 0, 0, 0)).asI should be (0)
    translation.valueAt(Array(1, 1, 0, 0)).asI should be (1)
    translation.valueAt(Array(1, 2, 0, 0)).asI should be (4)
    translation.valueAt(Array(2, 0, 0, 0)).asI should be (0)
    translation.valueAt(Array(2, 1, 0, 0)).asI should be (2)
    translation.valueAt(Array(2, 2, 0, 0)).asI should be (5)
  }

  "A 3x3x3 translation in the _X and _Z dimensions" should "have the expected magnitude, order, elementSize and element values" in {
    val inputTensor = ArrayTensor[T]((1 to 27).toArray.asTArray, Array(3, 3, 3), 0)
    val translation = TranslateTensor[T](inputTensor, Array(1, 0, 1), 0.asT)

    translation.magnitude should be (Array(3, 3, 3))
    translation.order should be (3)
    translation.elementSize should be (27)

    translation.valueAt(Array(0, 0, 0)).asI should be (0)
    translation.valueAt(Array(0, 0, 1)).asI should be (0)
    translation.valueAt(Array(0, 0, 2)).asI should be (0)

    translation.valueAt(Array(0, 1, 0)).asI should be (0)
    translation.valueAt(Array(0, 1, 1)).asI should be (0)
    translation.valueAt(Array(0, 1, 2)).asI should be (0)

    translation.valueAt(Array(0, 2, 0)).asI should be (0)
    translation.valueAt(Array(0, 2, 1)).asI should be (0)
    translation.valueAt(Array(0, 2, 2)).asI should be (0)

    translation.valueAt(Array(1, 0, 0)).asI should be (0)
    translation.valueAt(Array(1, 0, 1)).asI should be (1)
    translation.valueAt(Array(1, 0, 2)).asI should be (10)

    translation.valueAt(Array(1, 1, 0)).asI should be (0)
    translation.valueAt(Array(1, 1, 1)).asI should be (4)
    translation.valueAt(Array(1, 1, 2)).asI should be (13)

    translation.valueAt(Array(1, 2, 0)).asI should be (0)
    translation.valueAt(Array(1, 2, 1)).asI should be (7)
    translation.valueAt(Array(1, 2, 2)).asI should be (16)

    translation.valueAt(Array(2, 0, 0)).asI should be (0)
    translation.valueAt(Array(2, 0, 1)).asI should be (2)
    translation.valueAt(Array(2, 0, 2)).asI should be (11)

    translation.valueAt(Array(2, 1, 0)).asI should be (0)
    translation.valueAt(Array(2, 1, 1)).asI should be (5)
    translation.valueAt(Array(2, 1, 2)).asI should be (14)

    translation.valueAt(Array(2, 2, 0)).asI should be (0)
    translation.valueAt(Array(2, 2, 1)).asI should be (8)
    translation.valueAt(Array(2, 2, 2)).asI should be (17)
  }

  "A 3x3 tensor translated in the _Z dimension" should "have the expected magnitude, order, elementSize and element values" in {
    val inputTensor = ArrayTensor[T]((1 to 9).toArray.asTArray, Array(3, 3), 0)
    val translation = TranslateTensor[T](inputTensor, Array(0, 0, 1), 0.asT)

    translation.magnitude should be (Array(3, 3, 2))
    translation.order should be (3)
    translation.elementSize should be (18)

    translation.valueAt(Array(0, 0, 0)).asI should be (0)
    translation.valueAt(Array(0, 0, 1)).asI should be (1)

    translation.valueAt(Array(0, 1, 0)).asI should be (0)
    translation.valueAt(Array(0, 1, 1)).asI should be (4)

    translation.valueAt(Array(0, 2, 0)).asI should be (0)
    translation.valueAt(Array(0, 2, 1)).asI should be (7)

    translation.valueAt(Array(1, 0, 0)).asI should be (0)
    translation.valueAt(Array(1, 0, 1)).asI should be (2)

    translation.valueAt(Array(1, 1, 0)).asI should be (0)
    translation.valueAt(Array(1, 1, 1)).asI should be (5)

    translation.valueAt(Array(1, 2, 0)).asI should be (0)
    translation.valueAt(Array(1, 2, 1)).asI should be (8)

    translation.valueAt(Array(2, 0, 0)).asI should be (0)
    translation.valueAt(Array(2, 0, 1)).asI should be (3)

    translation.valueAt(Array(2, 1, 0)).asI should be (0)
    translation.valueAt(Array(2, 1, 1)).asI should be (6)

    translation.valueAt(Array(2, 2, 0)).asI should be (0)
    translation.valueAt(Array(2, 2, 1)).asI should be (9)
  }
}

class ByteTranslateTensorSpec extends TranslateTensorSpecBase[Byte]
class ShortTranslateTensorSpec extends TranslateTensorSpecBase[Short]
class IntTranslateTensorSpec extends TranslateTensorSpecBase[Int]
class LongTranslateTensorSpec extends TranslateTensorSpecBase[Long]
class FloatTranslateTensorSpec extends TranslateTensorSpecBase[Float]
class DoubleTranslateTensorSpec extends TranslateTensorSpecBase[Double]
