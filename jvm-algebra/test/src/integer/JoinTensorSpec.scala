package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

abstract class JoinTensorSpecBase[T: Numeric: ClassTag: ConvertsIntArray] extends TensorFlatSpecBase[T] {
  "Joining 2 2x3 tensors in the _X dimension" should "construct a tensor with expected magnitude, order, elementSize and element values" in {
    val t1 = ArrayTensor[T]((0 to 5).toArray.asTArray, Array(2, 3), 0)
    val t2 = ArrayTensor[T]((6 to 11).toArray.asTArray, Array(2, 3), 0)
    val join = JoinTensor[T](Array(t1, t2), _X)

    join.magnitude should be (Array(4, 3))
    join.order should be (2)
    join.elementSize should be (12)

    join.valueAt(Array(0, 0)).asI should be (0)
    join.valueAt(Array(1, 0)).asI should be (1)
    join.valueAt(Array(2, 0)).asI should be (6)
    join.valueAt(Array(3, 0)).asI should be (7)
    join.valueAt(Array(0, 1)).asI should be (2)
    join.valueAt(Array(1, 1)).asI should be (3)
    join.valueAt(Array(2, 1)).asI should be (8)
    join.valueAt(Array(3, 1)).asI should be (9)
    join.valueAt(Array(0, 2)).asI should be (4)
    join.valueAt(Array(1, 2)).asI should be (5)
    join.valueAt(Array(2, 2)).asI should be (10)
    join.valueAt(Array(3, 2)).asI should be (11)  }

  "Joining 2 2x3 tensors in the _Y dimension" should "construct a tensor with expected magnitude, order, elementSize and element values" in {
    val t1 = ArrayTensor[T]((0 to 5).toArray.asTArray, Array(2, 3), 0)
    val t2 = ArrayTensor[T]((6 to 11).toArray.asTArray, Array(2, 3), 0)
    val join = JoinTensor[T](Array(t1, t2), _Y)

    join.magnitude should be (Array(2, 6))
    join.order should be (2)
    join.elementSize should be (12)

    join.valueAt(Array(0, 0)).asI should be (0)
    join.valueAt(Array(1, 0)).asI should be (1)
    join.valueAt(Array(0, 1)).asI should be (2)
    join.valueAt(Array(1, 1)).asI should be (3)
    join.valueAt(Array(0, 2)).asI should be (4)
    join.valueAt(Array(1, 2)).asI should be (5)
    join.valueAt(Array(0, 3)).asI should be (6)
    join.valueAt(Array(1, 3)).asI should be (7)
    join.valueAt(Array(0, 4)).asI should be (8)
    join.valueAt(Array(1, 4)).asI should be (9)
    join.valueAt(Array(0, 5)).asI should be (10)
    join.valueAt(Array(1, 5)).asI should be (11)
  }
}

class ByteJoinTensorSpec extends JoinTensorSpecBase[Byte]
class ShortJoinTensorSpec extends JoinTensorSpecBase[Short]
class IntJoinTensorSpec extends JoinTensorSpecBase[Int]
class LongJoinTensorSpec extends JoinTensorSpecBase[Long]
class FloatJoinTensorSpec extends JoinTensorSpecBase[Float]
class DoubleJoinTensorSpec extends JoinTensorSpecBase[Double]
