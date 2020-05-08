package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

class ReduceTensorSpecBase[T: Numeric: ClassTag: ConvertsInt] extends TensorFlatSpecBase[T] {

  def sum(t: JVMTensor[T]): T = (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).sum

  "a 3x3 tensor reduced in 2 orders" should "yield a scalar tensor" in {
    val arrayTensor = ArrayTensor[T]((0 to 8).map(_ => 1).toArray.asTArray, Array(3, 3), 0)
    val reduce = ReduceTensor[T](arrayTensor, 2, sum)

    reduce.magnitude should be (Array(1))
    reduce.order should be (1)
    reduce.elementSize should be (1)

    reduce.valueAt(Array(0)).asI should be (9)
  }

  "a 3x3 tensor reduced in 1 order" should "yield a 1-D tensor with magnitude, order, elementSize and element values as expected" in {
    val arrayTensor = ArrayTensor[T]((0 to 8).toArray.asTArray, Array(3, 3), 0)
    val reduce = ReduceTensor[T](arrayTensor, 1, sum)

    reduce.magnitude should be (Array(3))
    reduce.order should be (1)
    reduce.elementSize should be (3)

    reduce.valueAt(Array(0)).asI should be (3)
    reduce.valueAt(Array(1)).asI should be (12)
    reduce.valueAt(Array(2)).asI should be (21)
  }

  "a 3x3 tensor reduced in 0 orders" should "yield a 2-D tensor with magnitude, order, elementSize and element values as expected" in {
    val arrayTensor = ArrayTensor[T]((0 to 8).toArray.asTArray, Array(3, 3), 0)
    val reduce = ReduceTensor[T](arrayTensor, 0, (t => implicitly[Numeric[T]].times(t.valueAt(Array(0)), 2.asT)))

    reduce.magnitude should be (Array(3, 3))
    reduce.order should be (2)
    reduce.elementSize should be (9)

    reduce.valueAt(Array(0, 0)).asI should be (0)
    reduce.valueAt(Array(1, 0)).asI should be (2)
    reduce.valueAt(Array(2, 0)).asI should be (4)
    reduce.valueAt(Array(0, 1)).asI should be (6)
    reduce.valueAt(Array(1, 1)).asI should be (8)
    reduce.valueAt(Array(2, 1)).asI should be (10)
    reduce.valueAt(Array(0, 2)).asI should be (12)
    reduce.valueAt(Array(1, 2)).asI should be (14)
    reduce.valueAt(Array(2, 2)).asI should be (16)
  }

  "a 2x3x4x3 tensor reduced in 2 orders" should "yield a 4x3 tensor with magnitude, order, elementSize and element values as expected" in {
    val arrayTensor = ArrayTensor[T]((0 to 71).map(_ => 1).toArray.asTArray, Array(2, 3, 4, 3), 0)
    val reduce = ReduceTensor[T](arrayTensor, 2, sum)

    reduce.magnitude should be (Array(4, 3))
    reduce.order should be (2)
    reduce.elementSize should be (12)

    reduce.valueAt(Array(0, 0)) should be (6)
    reduce.valueAt(Array(1, 0)) should be (6)
    reduce.valueAt(Array(0, 1)) should be (6)
    reduce.valueAt(Array(1, 1)) should be (6)
    reduce.valueAt(Array(0, 2)) should be (6)
    reduce.valueAt(Array(1, 2)) should be (6)
  }
}

class ByteReduceTensorSpec extends ReduceTensorSpecBase[Byte]
class ShortReduceTensorSpec extends ReduceTensorSpecBase[Short]
class IntReduceTensorSpec extends ReduceTensorSpecBase[Int]
class LongReduceTensorSpec extends ReduceTensorSpecBase[Long]
class FloatReduceTensorSpec extends ReduceTensorSpecBase[Float]
class DoubleReduceTensorSpec extends ReduceTensorSpecBase[Double]
