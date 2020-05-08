package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

class MapTensorSpecBase[T: Numeric: ClassTag: ConvertsInt] extends TensorFlatSpecBase[T] {
  "Mapping scalar tensor" should "construct a tensor with expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[T](Array[Int](5).asTArray, Array(1), 0)
    val mapTensor = MapTensor[T](arrayTensor, ((el) => numeric.times(el, 2.asT)))

    mapTensor.magnitude should be (Array(1))
    mapTensor.order should be (1)
    mapTensor.elementSize should be (1)

    mapTensor.valueAt(Array(0)).asI should be (10)
  }

  "Mapping a 2x3 tensor w/ _ * 2" should "construct a tensor with expected magnitude, order, elementSize and element values" in {
    val arrayTensor = ArrayTensor[T]((0 to 5).toArray.asTArray, Array(2, 3), 0)
    val mapTensor = MapTensor[T](arrayTensor, ((el) => numeric.times(el, 2.asT)))

    mapTensor.magnitude should be (Array(2, 3))
    mapTensor.order should be (2)
    mapTensor.elementSize should be (6)

    mapTensor.valueAt(Array(0, 0)).asI should be (0)
    mapTensor.valueAt(Array(1, 0)).asI should be (2)
    mapTensor.valueAt(Array(0, 1)).asI should be (4)
    mapTensor.valueAt(Array(1, 1)).asI should be (6)
    mapTensor.valueAt(Array(0, 2)).asI should be (8)
    mapTensor.valueAt(Array(1, 2)).asI should be (10)
  }
}

class ByteMapTensorSpec extends MapTensorSpecBase[Byte]
class ShortMapTensorSpec extends MapTensorSpecBase[Short]
class IntMapTensorSpec extends MapTensorSpecBase[Int]
class LongMapTensorSpec extends MapTensorSpecBase[Long]
class FloatMapTensorSpec extends MapTensorSpecBase[Float]
class DoubleMapTensorSpec extends MapTensorSpecBase[Double]
