package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

class ReverseTensorSpec extends FlatSpec {
  "A reverse tensor constructed from a 1D source tensor" should "have the expected magntude, order, elementSize and element values" in {
    val sourceTensor = ArrayTensor[Int]((0 to 5) toArray, Array(6), 0)
    val reversed = ReverseTensor(sourceTensor, _X)

    reversed.order should be (1)
    reversed.magnitude(_X) should be (6)
    reversed.elementSize should be (6)

    reversed.valueAt(Array(0)) should be (5)
    reversed.valueAt(Array(1)) should be (4)
    reversed.valueAt(Array(2)) should be (3)
    reversed.valueAt(Array(3)) should be (2)
    reversed.valueAt(Array(4)) should be (1)
    reversed.valueAt(Array(5)) should be (0)
  }

  "Reversing a tensor in a unitary dimension" should "have no effect" in {
    val sourceTensor = ArrayTensor[Int]((0 to 5) toArray, Array(6, 1), 0)
    val reversed = ReverseTensor(sourceTensor, _Y)

    reversed.order should be (2)
    reversed.magnitude(_X) should be (6)
    reversed.elementSize should be (6)

    reversed.valueAt(Array(0, 0)) should be (0)
    reversed.valueAt(Array(1, 0)) should be (1)
    reversed.valueAt(Array(2, 0)) should be (2)
    reversed.valueAt(Array(3, 0)) should be (3)
    reversed.valueAt(Array(4, 0)) should be (4)
    reversed.valueAt(Array(5, 0)) should be (5)
  }

  "Reversing a tensor in a higher dimension than the order in  a 3x3 tensor" should "have no effect" in {
    val sourceTensor = ArrayTensor[Int]((0 to 8) toArray, Array(3, 3), 0)
    val reversed = ReverseTensor(sourceTensor, _T)

    reversed.order should be (2)
    reversed.magnitude should be (Array(3, 3))
    reversed.elementSize should be (9)

    for(i <- 0 to 8) {
      reversed.valueAt1D(i) should be (i)
    }
  }

  "A tensor constructed by reversing an order-3 tensor in the _Y dimension" should "have the expected magnitude, order, elementSize and elementValues" in {
    val sourceTensor = ArrayTensor[Int]((0 to 11) toArray, Array(2, 3, 2), 0)
    val reversed = ReverseTensor(sourceTensor, _Y)

    reversed.order should be (3)
    reversed.magnitude should be (Array(2, 3, 2))
    reversed.elementSize should be (12)

    reversed.valueAt(Array(0, 0, 0)) should be (4)
    reversed.valueAt(Array(1, 0, 0)) should be (5)
    reversed.valueAt(Array(0, 1, 0)) should be (2)
    reversed.valueAt(Array(1, 1, 0)) should be (3)
    reversed.valueAt(Array(0, 2, 0)) should be (0)
    reversed.valueAt(Array(1, 2, 0)) should be (1)
    reversed.valueAt(Array(0, 0, 1)) should be (10)
    reversed.valueAt(Array(1, 0, 1)) should be (11)
    reversed.valueAt(Array(0, 1, 1)) should be (8)
    reversed.valueAt(Array(1, 1, 1)) should be (9)
    reversed.valueAt(Array(0, 2, 1)) should be (6)
    reversed.valueAt(Array(1, 2, 1)) should be (7)
  }
}
