package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

class ReshapeTensorSpec extends FlatSpec {
  "A 3x4x5 ReshapeTensor made from a 10x2x3 tensor" should "have the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = IntArrayTensor((0 to 59) toArray, Array(10, 2, 3), 0)
    val reshaped = ReshapeTensor(arrayTensor, Array(3, 4, 5))

    reshaped.magnitude should be (Array(3, 4, 5))
    reshaped.order should be (3)
    reshaped.elementSize should be (60)

    reshaped.valueAt(Array(0, 0, 0)) should be (0)
    reshaped.valueAt(Array(2, 2, 1)) should be (20)
    reshaped.valueAt(Array(0, 1, 4)) should be (51)
    reshaped.valueAt(Array(1, 3, 2)) should be (34)
    reshaped.valueAt(Array(2, 3, 4)) should be (59)
  }

  "Providing additional dimension index 0 values when getting values from a ReshapedTensor" should "yield the same as not providing those values" in {
    val arrayTensor = IntArrayTensor((0 to 59) toArray, Array(10, 2, 3), 0)
    val reshaped = ReshapeTensor(arrayTensor, Array(3, 4, 5))

    reshaped.valueAt(Array(0, 0, 0)) should be (reshaped.valueAt(Array(0, 0, 0, 0)))
    reshaped.valueAt(Array(2, 2, 1)) should be (reshaped.valueAt(Array(2, 2, 1, 0, 0)))
    reshaped.valueAt(Array(0, 1, 4)) should be (reshaped.valueAt(Array(0, 1, 4, 0, 0, 0)))
    reshaped.valueAt(Array(1, 3, 2)) should be (reshaped.valueAt(Array(1, 3, 2, 0)))
    reshaped.valueAt(Array(2, 3, 4)) should be (reshaped.valueAt(Array(2, 3, 4, 0, 0, 0, 0)))
  }
}
