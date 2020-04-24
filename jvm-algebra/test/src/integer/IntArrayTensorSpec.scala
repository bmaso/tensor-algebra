package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

class IntArrayTensorSpec extends FlatSpec {
  "A 5x5 tensor" should "have expected magnitude, order, and elementSize" in {
    val t = IntArrayTensor((0 to 24) toArray, Array(5, 5), 0)
    t.order should be (2)
    t.magnitude(_X) should be (5)
    t.magnitude(_Y) should be (5)
    t.elementSize should be (25)
  }

  it should "yield expected values from valueAt" in {
    val t = IntArrayTensor((0 to 24) toArray, Array(5, 5), 0)
    t.valueAt(Array(0, 0)) should be (0)
    t.valueAt(Array(2, 0)) should be (2)
    t.valueAt(Array(3, 3)) should be (18)
    t.valueAt(Array(3, 1)) should be (8)
    t.valueAt(Array(4, 4)) should be (24)
  }

  it should "yield expected values from valueAt1D" in {
    val t = IntArrayTensor((0 to 24) toArray, Array(5, 5), 0)
    t.valueAt1D(0) should be (0)
    t.valueAt1D(10) should be (10)
    t.valueAt1D(18) should be (18)
    t.valueAt1D(16) should be (16)
    t.valueAt1D(24) should be (24)
  }

  "A 7x3x8x2 tensor" should "have expected values from valueAt" in {
    val t = IntArrayTensor((0 to 335) toArray, Array(7, 3, 8, 2), 0)
    t.valueAt(Array(0, 0, 0, 0)) should be (0)
    t.valueAt(Array(2, 0, 0, 0)) should be (2)
    t.valueAt(Array(4, 2, 5, 1)) should be (291) //  4 + 14 + 105 + 168
    t.valueAt(Array(6, 2, 7, 1)) should be (335) //  6 + 14 + 147 + 168
  }

  it should "yield expected values from valueAt1D" in {
    val t = IntArrayTensor((0 to 335) toArray, Array(7, 3, 8, 2), 0)
    t.valueAt1D(0) should be (0)
    t.valueAt1D(96) should be (96)
    t.valueAt1D(235) should be (235)
    t.valueAt1D(335) should be (335)
  }

  "An attempt to create a tensor with magnitude/length mismatch" should "throw IllegalArgumentException" in {
    intercept[IllegalArgumentException] {
      IntArrayTensor((1 to 100) toArray, Array(5, 25), 0)
    }
  }

  "a tensor" should "ignore additional unitary values in index passed to valueAt" in {
    val t = IntArrayTensor((0 to 63) toArray, Array(4, 4, 4), 0)

    t.valueAt(Array(0, 0, 0)) should be (0)
    t.valueAt(Array(0, 0, 0, 0, 0, 0)) should be (0)

    t.valueAt(Array(1, 1, 1)) should be (21)
    t.valueAt(Array(1, 1, 1, 0, 0, 0)) should be (21)
  }
}
