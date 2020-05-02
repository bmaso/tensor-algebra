package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

class BroadcastTensorSpec extends FlatSpec {
  import IntTensorAlgebra._

  "A 1x2x3 tensor broadcast by 2 in _X" should "yield a 2x2x3 tensor with the expected magnitude, order, elementSize and element values" in {
    val arrayTensor = IntArrayTensor((0 to 5) toArray, Array(1, 2, 3), 0)
    val broadcastTensor = BroadcastTensor(arrayTensor, _X, 2)

    broadcastTensor.magnitude should be (Array(2, 2, 3))
    broadcastTensor.order should be (3)
    broadcastTensor.elementSize should be (12)

    broadcastTensor.valueAt(Array(0, 0, 0)) should be (0)
    broadcastTensor.valueAt(Array(1, 0, 0)) should be (0)
    broadcastTensor.valueAt(Array(0, 1, 0)) should be (1)
    broadcastTensor.valueAt(Array(1, 1, 0)) should be (1)
    broadcastTensor.valueAt(Array(0, 0, 1)) should be (2)
    broadcastTensor.valueAt(Array(1, 0, 1)) should be (2)
    broadcastTensor.valueAt(Array(0, 1, 1)) should be (3)
    broadcastTensor.valueAt(Array(1, 1, 1)) should be (3)
    broadcastTensor.valueAt(Array(0, 0, 2)) should be (4)
    broadcastTensor.valueAt(Array(1, 0, 2)) should be (4)
    broadcastTensor.valueAt(Array(0, 1, 2)) should be (5)
    broadcastTensor.valueAt(Array(1, 1, 2)) should be (5)
  }
}
