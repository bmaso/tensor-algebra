package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions.{TensorAlgebra => abstract_TensorAlgebra, _}

abstract class BroadcastTensorSpecBase[T: Numeric: ClassTag: ConvertsIntArray] extends FlatSpec {
  val numeric: Numeric[T] = implicitly[Numeric[T]]

  implicit def canConvertToInt(t: T) = new {
    def asI: Int = numeric.toInt(t)
  }

  implicit def canConvertIntArray(array: Array[Int]) = new {
    def asTArray: Array[T] = {
      implicitly[ConvertsIntArray[T]].convertIntArray(array)
    }
  }

  "A 1x2x3 tensor broadcast by 2 in _X" should "yield a 2x2x3 tensor with the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[T](((0 to 5) toArray).asTArray, Array(1, 2, 3), 0)
    val broadcastTensor = BroadcastTensor(arrayTensor, _X, 2)

    broadcastTensor.magnitude should be (Array(2, 2, 3))
    broadcastTensor.order should be (3)
    broadcastTensor.elementSize should be (12)

    broadcastTensor.valueAt(Array(0, 0, 0)).asI should be (0)
    broadcastTensor.valueAt(Array(1, 0, 0)).asI should be (0)
    broadcastTensor.valueAt(Array(0, 1, 0)).asI should be (1)
    broadcastTensor.valueAt(Array(1, 1, 0)).asI should be (1)
    broadcastTensor.valueAt(Array(0, 0, 1)).asI should be (2)
    broadcastTensor.valueAt(Array(1, 0, 1)).asI should be (2)
    broadcastTensor.valueAt(Array(0, 1, 1)).asI should be (3)
    broadcastTensor.valueAt(Array(1, 1, 1)).asI should be (3)
    broadcastTensor.valueAt(Array(0, 0, 2)).asI should be (4)
    broadcastTensor.valueAt(Array(1, 0, 2)).asI should be (4)
    broadcastTensor.valueAt(Array(0, 1, 2)).asI should be (5)
    broadcastTensor.valueAt(Array(1, 1, 2)).asI should be (5)
  }

  "A 2x1x3 tensor broadcast by 2 in _Y" should "yield a 2x2x3 tensor with the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[T](((0 to 5) toArray).asTArray, Array(2, 1, 3), 0)
    val broadcastTensor = BroadcastTensor(arrayTensor, _Y, 2)

    broadcastTensor.magnitude should be (Array(2, 2, 3))
    broadcastTensor.order should be (3)
    broadcastTensor.elementSize should be (12)

    broadcastTensor.valueAt(Array(0, 0, 0)).asI should be (0)
    broadcastTensor.valueAt(Array(1, 0, 0)).asI should be (1)
    broadcastTensor.valueAt(Array(0, 1, 0)).asI should be (0)
    broadcastTensor.valueAt(Array(1, 1, 0)).asI should be (1)
    broadcastTensor.valueAt(Array(0, 0, 1)).asI should be (2)
    broadcastTensor.valueAt(Array(1, 0, 1)).asI should be (3)
    broadcastTensor.valueAt(Array(0, 1, 1)).asI should be (2)
    broadcastTensor.valueAt(Array(1, 1, 1)).asI should be (3)
    broadcastTensor.valueAt(Array(0, 0, 2)).asI should be (4)
    broadcastTensor.valueAt(Array(1, 0, 2)).asI should be (5)
    broadcastTensor.valueAt(Array(0, 1, 2)).asI should be (4)
    broadcastTensor.valueAt(Array(1, 1, 2)).asI should be (5)
  }

  "A 2x3 tensor broadcast by 2 in _W" should "yield a 2x3x1x2 tensor with the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[T](((0 to 5) toArray).asTArray, Array(2, 3), 0)
    val broadcastTensor = BroadcastTensor(arrayTensor, _W, 2)

    broadcastTensor.magnitude should be (Array(2, 3, 1, 2))
    broadcastTensor.order should be (4)
    broadcastTensor.elementSize should be (12)

    broadcastTensor.valueAt(Array(0, 0, 0, 0)).asI should be (0)
    broadcastTensor.valueAt(Array(1, 0, 0, 0)).asI should be (1)
    broadcastTensor.valueAt(Array(0, 1, 0, 0)).asI should be (2)
    broadcastTensor.valueAt(Array(1, 1, 0, 0)).asI should be (3)
    broadcastTensor.valueAt(Array(0, 2, 0, 0)).asI should be (4)
    broadcastTensor.valueAt(Array(1, 2, 0, 0)).asI should be (5)
    broadcastTensor.valueAt(Array(0, 0, 0, 1)).asI should be (0)
    broadcastTensor.valueAt(Array(1, 0, 0, 1)).asI should be (1)
    broadcastTensor.valueAt(Array(0, 1, 0, 1)).asI should be (2)
    broadcastTensor.valueAt(Array(1, 1, 0, 1)).asI should be (3)
    broadcastTensor.valueAt(Array(0, 2, 0, 1)).asI should be (4)
    broadcastTensor.valueAt(Array(1, 2, 0, 1)).asI should be (5)
  }
}

class ByteBroadcastTensorSpec extends BroadcastTensorSpecBase[Byte]
class ShortBroadcastTensorSpec extends BroadcastTensorSpecBase[Short]
class IntBroadcastTensorSpec extends BroadcastTensorSpecBase[Int]
class LongBroadcastTensorSpec extends BroadcastTensorSpecBase[Long]
class FloatBroadcastTesnorSpec extends BroadcastTensorSpecBase[Float]
class DoubleBroadcastTesnorSpec extends BroadcastTensorSpecBase[Double]
