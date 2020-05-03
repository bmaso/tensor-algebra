package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

class ReduceTensorSpec extends FlatSpec {

  def sum(t: JVMTensor[Int]): Int = (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).sum

  "a 3x3 tensor reduced in 2 orders" should "yield a scalar tensor" in {
    import IntTensorAlgebra._

    val arrayTensor = ArrayTensor[Int]((0 to 8).map(_ => 1) toArray, Array(3, 3), 0)
    val reduce = ReduceTensor[Int](arrayTensor, 2, sum)

    reduce.magnitude should be (Array(1))
    reduce.order should be (1)
    reduce.elementSize should be (1)

    reduce.valueAt(Array(0)) should be (9)
  }

  "a 3x3 tensor reduced in 1 order" should "yield a 1-D tensor with magnitude, order, elementSize and element values as expected" in {
    import IntTensorAlgebra._

    val arrayTensor = ArrayTensor[Int]((0 to 8) toArray, Array(3, 3), 0)
    val reduce = ReduceTensor[Int](arrayTensor, 1, sum)

    reduce.magnitude should be (Array(3))
    reduce.order should be (1)
    reduce.elementSize should be (3)

    reduce.valueAt(Array(0)) should be (3)
    reduce.valueAt(Array(1)) should be (12)
    reduce.valueAt(Array(2)) should be (21)
  }

  "a 3x3 tensor reduced in 0 orders" should "yield a 2-D tensor with magnitude, order, elementSize and element values as expected" in {
    import IntTensorAlgebra._

    val arrayTensor = ArrayTensor[Int]((0 to 8) toArray, Array(3, 3), 0)
    val reduce = ReduceTensor[Int](arrayTensor, 0, (t => t.valueAt(Array(0)) * 2))

    reduce.magnitude should be (Array(3, 3))
    reduce.order should be (2)
    reduce.elementSize should be (9)

    reduce.valueAt(Array(0, 0)) should be (0)
    reduce.valueAt(Array(1, 0)) should be (2)
    reduce.valueAt(Array(2, 0)) should be (4)
    reduce.valueAt(Array(0, 1)) should be (6)
    reduce.valueAt(Array(1, 1)) should be (8)
    reduce.valueAt(Array(2, 1)) should be (10)
    reduce.valueAt(Array(0, 2)) should be (12)
    reduce.valueAt(Array(1, 2)) should be (14)
    reduce.valueAt(Array(2, 2)) should be (16)
  }

  "a 6x5x4x3 tensor reduced in 2 orders" should "yield a 4x3 tensor with magnitude, order, elementSize and element values as expected" in {
    import IntTensorAlgebra._

    val arrayTensor = ArrayTensor[Int]((0 to 359).map(_ => 1) toArray, Array(6, 5, 4, 3), 0)
    val reduce = ReduceTensor[Int](arrayTensor, 2, sum)

    reduce.magnitude should be (Array(4, 3))
    reduce.order should be (2)
    reduce.elementSize should be (12)

    reduce.valueAt(Array(0, 0)) should be (30)
    reduce.valueAt(Array(1, 0)) should be (30)
    reduce.valueAt(Array(2, 0)) should be (30)
    reduce.valueAt(Array(3, 0)) should be (30)
    reduce.valueAt(Array(0, 1)) should be (30)
    reduce.valueAt(Array(1, 1)) should be (30)
    reduce.valueAt(Array(2, 1)) should be (30)
    reduce.valueAt(Array(3, 1)) should be (30)
    reduce.valueAt(Array(0, 2)) should be (30)
    reduce.valueAt(Array(1, 2)) should be (30)
    reduce.valueAt(Array(2, 2)) should be (30)
    reduce.valueAt(Array(3, 2)) should be (30)
  }
}
