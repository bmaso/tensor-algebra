package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

class MapTensorSpec extends FlatSpec {
  "Mapping scalar tensor" should "construct a tensor with expected magnitude, order, elementSize and element values" in {
    val arrayTensor = IntArrayTensor(Array[Int](5), Array(1), 0)
    val mapTensor = MapTensor(arrayTensor, (_ * 2))

    mapTensor.magnitude should be (Array(1))
    mapTensor.order should be (1)
    mapTensor.elementSize should be (1)

    mapTensor.valueAt(Array(0)) should be (10)
  }

  "Mapping a 2x3 tensor w/ _ * 2" should "construct a tensor with expected magnitude, order, elementSize and element values" in {
    val arrayTensor = IntArrayTensor((0 to 5) toArray, Array(2, 3), 0)
    val mapTensor = MapTensor(arrayTensor, (_ * 2))

    mapTensor.magnitude should be (Array(2, 3))
    mapTensor.order should be (2)
    mapTensor.elementSize should be (6)

    mapTensor.valueAt(Array(0, 0)) should be (0)
    mapTensor.valueAt(Array(1, 0)) should be (2)
    mapTensor.valueAt(Array(0, 1)) should be (4)
    mapTensor.valueAt(Array(1, 1)) should be (6)
    mapTensor.valueAt(Array(0, 2)) should be (8)
    mapTensor.valueAt(Array(1, 2)) should be (10)
  }
}
