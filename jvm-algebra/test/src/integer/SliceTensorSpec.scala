package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._


class SliceTensorSpec extends FlatSpec{
  "A 2x2x2 slice of a 4x4x4 tensor" should "have the right magnitude, order, elementSize, and element values" in {
    val arrayTensor = IntArrayTensor((0 to 63) toArray, Array(4, 4, 4), 0)
    val slice = SliceTensor(arrayTensor, Array((1, 2), (1, 2), (1, 2)))

    slice.magnitude should be (Array(2, 2, 2))
    slice.order should be (3)
    slice.elementSize should be (8)

    slice.valueAt(Array(0, 0, 0)) should be (21)
    slice.valueAt(Array(1, 0, 1)) should be (38)
    slice.valueAt(Array(0, 1, 0)) should be (25)
  }

  "A 2x2x1 slice of a 4x4x4 tensor" should "have the right magnitude, order, elementSize, and element values" in {
    val arrayTensor = IntArrayTensor((0 to 63) toArray, Array(4, 4, 4), 0)
    val slice = SliceTensor(arrayTensor, Array((1, 2), (1, 2), (1, 1)))

    slice.magnitude should be (Array(2, 2))
    slice.order should be (2)
    slice.elementSize should be (4)

    slice.valueAt(Array(0, 0)) should be (21)
    slice.valueAt(Array(1, 0)) should be (22)
    slice.valueAt(Array(0, 1)) should be (25)
    slice.valueAt(Array(1, 1)) should be (26)
  }

  "A 2x1x2 slice of a 4x4x4 tensor" should "have the right magnitude, order, elementSize, and element values" in {
    val arrayTensor = IntArrayTensor((0 to 63) toArray, Array(4, 4, 4), 0)
    val slice = SliceTensor(arrayTensor, Array((1, 2), (1, 1), (1, 2)))

    slice.magnitude should be (Array(2, 1, 2))
    slice.order should be (3)
    slice.elementSize should be (4)

    slice.valueAt(Array(0, 0, 0)) should be (21)
    slice.valueAt(Array(1, 0, 0)) should be (22)
    slice.valueAt(Array(0, 0, 1)) should be (37)
    slice.valueAt(Array(1, 0, 1)) should be (38)
  }
}
