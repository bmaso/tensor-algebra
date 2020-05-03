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
}

class ShortBroadcastTensorSpec extends BroadcastTensorSpecBase[Short]
class IntBroadcastTensorSpec extends BroadcastTensorSpecBase[Int]
class LongBroadcastTensorSpec extends BroadcastTensorSpecBase[Long]
