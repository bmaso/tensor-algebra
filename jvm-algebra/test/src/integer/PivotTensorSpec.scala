package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

class PivotTensorSpec extends FlatSpec {
  "A tensor constructed as a (_X, _Y) pivot of a 3x2 tensor" should "have the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[Int]((0 to 5) toArray, Array(3, 2), 0)
    val pivot = PivotTensor(arrayTensor, _X, _Y)

    pivot.magnitude should be (Array(2, 3))
    pivot.order should be (2)
    pivot.elementSize should be (6)

    pivot.valueAt(Array(0, 0)) should be (0)
    pivot.valueAt(Array(1, 0)) should be (3)
    pivot.valueAt(Array(0, 1)) should be (1)
    pivot.valueAt(Array(1, 1)) should be (4)
    pivot.valueAt(Array(0, 2)) should be (2)
    pivot.valueAt(Array(1, 2)) should be (5)
  }

  "A tensor constructed as a (_X, _Z) pivot of a 3x2 tensor" should "have the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[Int]((0 to 5) toArray, Array(3, 2), 0)
    val pivot = PivotTensor(arrayTensor, _X, _Z)

    pivot.magnitude should be (Array(1, 2, 3))
    pivot.order should be (3)
    pivot.elementSize should be (6)

    pivot.valueAt(Array(0, 0, 0)) should be (0)
    pivot.valueAt(Array(0, 1, 0)) should be (3)
    pivot.valueAt(Array(0, 0, 1)) should be (1)
    pivot.valueAt(Array(0, 1, 1)) should be (4)
    pivot.valueAt(Array(0, 0, 2)) should be (2)
    pivot.valueAt(Array(0, 1, 2)) should be (5)
  }

  "A tensor constructed as a (_X, _X) pivot of a 3x2 tensor" should "be value-wise identical to the original" in {
    val arrayTensor = ArrayTensor[Int]((0 to 5) toArray, Array(3, 2), 0)
    val pivot = PivotTensor(arrayTensor, _X, _X)

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
