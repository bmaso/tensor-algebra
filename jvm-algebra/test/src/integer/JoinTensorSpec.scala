package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

class JoinTensorSpec extends FlatSpec {
  "Joining 2 2x3 tensors in the _X dimension" should "construct a tensor with expected magnitude, order, elementSize and element values" in {
    val t1 = ArrayTensor[Int]((0 to 5) toArray, Array(2, 3), 0)
    val t2 = ArrayTensor[Int]((6 to 11) toArray, Array(2, 3), 0)
    val join = JoinTensor[Int](Array(t1, t2), _X)

    join.magnitude should be (Array(4, 3))
    join.order should be (2)
    join.elementSize should be (12)

    join.valueAt(Array(0, 0)) should be (0)
    join.valueAt(Array(1, 0)) should be (1)
    join.valueAt(Array(2, 0)) should be (6)
    join.valueAt(Array(3, 0)) should be (7)
    join.valueAt(Array(0, 1)) should be (2)
    join.valueAt(Array(1, 1)) should be (3)
    join.valueAt(Array(2, 1)) should be (8)
    join.valueAt(Array(3, 1)) should be (9)
    join.valueAt(Array(0, 2)) should be (4)
    join.valueAt(Array(1, 2)) should be (5)
    join.valueAt(Array(2, 2)) should be (10)
    join.valueAt(Array(3, 2)) should be (11)
  }

  "Joining 2 2x3 tensors in the _Y dimension" should "construct a tensor with expected magnitude, order, elementSize and element values" in {
    val t1 = ArrayTensor[Int]((0 to 5) toArray, Array(2, 3), 0)
    val t2 = ArrayTensor[Int]((6 to 11) toArray, Array(2, 3), 0)
    val join = JoinTensor[Int](Array(t1, t2), _Y)

    join.magnitude should be (Array(2, 6))
    join.order should be (2)
    join.elementSize should be (12)

    join.valueAt(Array(0, 0)) should be (0)
    join.valueAt(Array(1, 0)) should be (1)
    join.valueAt(Array(0, 1)) should be (2)
    join.valueAt(Array(1, 1)) should be (3)
    join.valueAt(Array(0, 2)) should be (4)
    join.valueAt(Array(1, 2)) should be (5)
    join.valueAt(Array(0, 3)) should be (6)
    join.valueAt(Array(1, 3)) should be (7)
    join.valueAt(Array(0, 4)) should be (8)
    join.valueAt(Array(1, 4)) should be (9)
    join.valueAt(Array(0, 5)) should be (10)
    join.valueAt(Array(1, 5)) should be (11)
  }
}
