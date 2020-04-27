package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

class StackTensorSpec extends FlatSpec {
  "A stack of 2 3x2 tensors in the _Z dimension" should "have the expected magnitude, order, elementSize and elemetn values" in {
    import IntTensorAlgebra._

    val t1 = IntArrayTensor((0 to 5) toArray, Array(3, 2), 0)
    val t2 = IntArrayTensor((6 to 11) toArray, Array(3, 2), 0)
    val stack = StackTensor(Array(t1, t2), _Z)

    stack.magnitude should be (Array(3, 2, 2))
    stack.order should be (3)
    stack.elementSize should be (12)

    for(ii <- 0 to 11) {
      stack.valueAt1D(ii) should be (ii)
    }

    //...a couple index-based element size checks to make sure allis well...
    stack.valueAt(Array(2, 1, 1)) should be (11)
    stack.valueAt(Array(0, 1, 0)) should be (3)
    stack.valueAt(Array(1, 0, 1)) should be (7)
  }

  "A stack of 2 1x2x3 tensors in the _X direction" should "have the expected magnitude, order, elementSize and elemetn values" in {
    val t1 = IntArrayTensor((0 to 5) toArray, Array(1, 2, 3), 0)
    val t2 = IntArrayTensor((6 to 11) toArray, Array(1, 2, 3), 0)
    val stack = StackTensor(Array(t1, t2), _X)

    stack.magnitude should be (Array(2, 2, 3))
    stack.order should be (3)
    stack.elementSize should be (12)

    stack.valueAt(Array(0, 0, 0)) should be (0)
    stack.valueAt(Array(1, 0, 0)) should be (6)
    stack.valueAt(Array(0, 1, 0)) should be (1)
    stack.valueAt(Array(1, 1, 0)) should be (7)
    stack.valueAt(Array(0, 0, 1)) should be (2)
    stack.valueAt(Array(1, 0, 1)) should be (8)
    stack.valueAt(Array(0, 1, 1)) should be (3)
    stack.valueAt(Array(1, 1, 1)) should be (9)
    stack.valueAt(Array(0, 0, 2)) should be (4)
    stack.valueAt(Array(1, 0, 2)) should be (10)
    stack.valueAt(Array(0, 1, 2)) should be (5)
    stack.valueAt(Array(1, 1, 2)) should be (11)
  }
}
