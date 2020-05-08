package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

abstract class PivotTensorSpecBase[T: Numeric: ClassTag: ConvertsInt] extends TensorFlatSpecBase[T] {
  "A tensor constructed as a (_X, _Y) pivot of a 3x2 tensor" should "have the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[T]((0 to 5).toArray.asTArray, Array(3, 2), 0)
    val pivot = PivotTensor[T](arrayTensor, _X, _Y)

    pivot.magnitude should be (Array(2, 3))
    pivot.order should be (2)
    pivot.elementSize should be (6)

    pivot.valueAt(Array(0, 0)).asI should be (0)
    pivot.valueAt(Array(1, 0)).asI should be (3)
    pivot.valueAt(Array(0, 1)).asI should be (1)
    pivot.valueAt(Array(1, 1)).asI should be (4)
    pivot.valueAt(Array(0, 2)).asI should be (2)
    pivot.valueAt(Array(1, 2)).asI should be (5)
  }

  "A tensor constructed as a (_X, _Z) pivot of a 3x2 tensor" should "have the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[T]((0 to 5).toArray.asTArray, Array(3, 2), 0)
    val pivot = PivotTensor[T](arrayTensor, _X, _Z)

    pivot.magnitude should be (Array(1, 2, 3))
    pivot.order should be (3)
    pivot.elementSize should be (6)

    pivot.valueAt(Array(0, 0, 0)).asI should be (0)
    pivot.valueAt(Array(0, 1, 0)).asI should be (3)
    pivot.valueAt(Array(0, 0, 1)).asI should be (1)
    pivot.valueAt(Array(0, 1, 1)).asI should be (4)
    pivot.valueAt(Array(0, 0, 2)).asI should be (2)
    pivot.valueAt(Array(0, 1, 2)).asI should be (5)
  }

  "A tensor constructed as a (_X, _X) pivot of a 3x2 tensor" should "be value-wise identical to the original" in {
    val arrayTensor = ArrayTensor[T]((0 to 5).toArray.asTArray, Array(3, 2), 0)
    val pivot = PivotTensor[T](arrayTensor, _X, _X)

    pivot.magnitude should be (arrayTensor.magnitude)
    pivot.order should be (arrayTensor.order)
    pivot.elementSize should be (arrayTensor.elementSize)

    pivot.valueAt(Array(0, 0)) should be (arrayTensor.valueAt(Array(0, 0)))
    pivot.valueAt(Array(1, 0)) should be (arrayTensor.valueAt(Array(1, 0)))
    pivot.valueAt(Array(2, 0)) should be (arrayTensor.valueAt(Array(2, 0)))
    pivot.valueAt(Array(0, 1)) should be (arrayTensor.valueAt(Array(0, 1)))
    pivot.valueAt(Array(1, 1)) should be (arrayTensor.valueAt(Array(1, 1)))
    pivot.valueAt(Array(2, 1)) should be (arrayTensor.valueAt(Array(2, 1)))
  }
}

class BytePivotTensorSpec extends PivotTensorSpecBase[Byte]
class ShortPivotTensorSpec extends PivotTensorSpecBase[Short]
class IntPivotTensorSpec extends PivotTensorSpecBase[Int]
class LongPivotTensorSpec extends PivotTensorSpecBase[Long]
class FloatPivotTensorSpec extends PivotTensorSpecBase[Float]
class DoublePivotTensorSpec extends PivotTensorSpecBase[Double]
