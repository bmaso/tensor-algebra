package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

class StackTensorSpecBase[T: Numeric: ClassTag: ConvertsInt] extends TensorFlatSpecBase[T] {
  "A stack of 2 3x2 tensors in the _Z dimension" should "have the expected magnitude, order, elementSize and elemetn values" in {
    val t1 = ArrayTensor[T]((0 to 5).toArray.asTArray, Array(3, 2), 0)
    val t2 = ArrayTensor[T]((6 to 11).toArray.asTArray, Array(3, 2), 0)
    val stack = StackTensor[T](Array(t1, t2), _Z)

    stack.magnitude should be (Array(3, 2, 2))
    stack.order should be (3)
    stack.elementSize should be (12)

    for(ii <- 0 to 11) {
      stack.valueAt1D(ii).asI should be (ii)
    }

    //...a couple index-based element size checks to make sure allis well...
    stack.valueAt(Array(2, 1, 1)).asI should be (11)
    stack.valueAt(Array(0, 1, 0)).asI should be (3)
    stack.valueAt(Array(1, 0, 1)).asI should be (7)
  }

  "A stack of 2 1x2x3 tensors in the _X direction" should "have the expected magnitude, order, elementSize and elemetn values" in {
    val t1 = ArrayTensor[T]((0 to 5).toArray.asTArray, Array(1, 2, 3), 0)
    val t2 = ArrayTensor[T]((6 to 11).toArray.asTArray, Array(1, 2, 3), 0)
    val stack = StackTensor[T](Array(t1, t2), _X)

    stack.magnitude should be (Array(2, 2, 3))
    stack.order should be (3)
    stack.elementSize should be (12)

    stack.valueAt(Array(0, 0, 0)).asI should be (0)
    stack.valueAt(Array(1, 0, 0)).asI should be (6)
    stack.valueAt(Array(0, 1, 0)).asI should be (1)
    stack.valueAt(Array(1, 1, 0)).asI should be (7)
    stack.valueAt(Array(0, 0, 1)).asI should be (2)
    stack.valueAt(Array(1, 0, 1)).asI should be (8)
    stack.valueAt(Array(0, 1, 1)).asI should be (3)
    stack.valueAt(Array(1, 1, 1)).asI should be (9)
    stack.valueAt(Array(0, 0, 2)).asI should be (4)
    stack.valueAt(Array(1, 0, 2)).asI should be (10)
    stack.valueAt(Array(0, 1, 2)).asI should be (5)
    stack.valueAt(Array(1, 1, 2)).asI should be (11)
  }
}

class ByteStackTensorSpec extends StackTensorSpecBase[Byte]
class ShortStackTensorSpec extends StackTensorSpecBase[Short]
class IntStackTensorSpec extends StackTensorSpecBase[Int]
class LongStackTensorSpec extends StackTensorSpecBase[Long]
class FloatStackTensorSpec extends StackTensorSpecBase[Float]
class DoubleStackTensorSpec extends StackTensorSpecBase[Double]
