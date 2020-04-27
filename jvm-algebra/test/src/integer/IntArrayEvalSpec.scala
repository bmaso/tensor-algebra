package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions._

/**
 * This test assures that the JVM Int evaluator correctly implements the
 * common abstract operations by putting them through their paces:
 * * tensorFromArray
 * * copyTensorElementsToArray
 **/
class IntArrayEvalSpec extends FlatSpec {
  "An eval of Copying array to array" should "transfer array contents unchanged" in {
    val magnitude = Array(5, 5)
    val inputArray = (0 to 24) toArray
    val outputArray = Array.fill[Int](25)(0)

    val expr = IntTensorAlgebra.tensorFromArray(inputArray, Array(5, 5))
        .flatMap(tensor => IntTensorAlgebra.copyTensorElementsToArray(tensor, outputArray))

    val interp = IdInterpreter

    val _ = interp.eval(expr)

    outputArray should be (inputArray)
  }

  "An eval of translating 3x3 array (1, 1)" should "yield shifted elements backfilled with 0s" in {
    import IntTensorAlgebra._

    val inputArray = (1 to 9) toArray
    val outputArray = Array.fill[Int](9)(0)

    val expr = tensorFromArray(inputArray, Array(3, 3))
      .flatMap(translate(_, Array(1, 1)))
      .flatMap(copyTensorElementsToArray(_, outputArray))

    val interp = IdInterpreter
    val _ = interp.eval(expr)

    outputArray should be (Array(0, 0, 0,
                                 0, 1, 2,
                                 0, 4, 5))
  }

  "An eval of broadcasting a 4x2 tensor with 5x3 base magnitude" should "yield a tensor with magnitude, order, elementSize, and element values as expected" in {
    import IntTensorAlgebra._

    val inputArray = Array(1, 3, 5, 7,
                           2, 4, 6, 8)

    val expr = tensorFromArray(inputArray, Array(4, 2))
      .flatMap(broadcast(_, Array(5, 3)))
      .flatMap({t =>
        t.magnitude should be (Array(5, 3, 4, 2))
        t.order should be (4)
        t.elementSize should be (120)

        //...test a few random elements to make sure broadcast projects the input tensor as expected...
        t.valueAt(Array(0, 0, 0, 0)) should be (1)
        t.valueAt(Array(0, 0, 1, 0)) should be (3)
        t.valueAt(Array(0, 0, 2, 0)) should be (5)
        t.valueAt(Array(0, 0, 3, 0)) should be (7)
        t.valueAt(Array(0, 0, 0, 1)) should be (2)
        t.valueAt(Array(0, 0, 1, 1)) should be (4)
        t.valueAt(Array(0, 0, 2, 1)) should be (6)
        t.valueAt(Array(0, 0, 3, 1)) should be (8)

        t.valueAt(Array(4, 2, 0, 0)) should be (1)
        t.valueAt(Array(3, 1, 1, 0)) should be (3)
        t.valueAt(Array(2, 0, 2, 0)) should be (5)
        t.valueAt(Array(1, 2, 3, 0)) should be (7)
        t.valueAt(Array(0, 1, 0, 1)) should be (2)
        t.valueAt(Array(4, 0, 1, 1)) should be (4)
        t.valueAt(Array(3, 2, 2, 1)) should be (6)
        t.valueAt(Array(2, 1, 3, 1)) should be (8)
        t.valueAt(Array(1, 0, 0, 0)) should be (1)
        t.valueAt(Array(0, 2, 3, 1)) should be (8)

        unit()
      })

    val interp = IdInterpreter
    val _ = interp.eval(expr)

    succeed
  }

  "Eval of slicing a 4x4x4 tensor" should "produce tensor with expected magnitude, order, elementSize and element values" in {
    import IntTensorAlgebra._

    val inputArray = (0 to 63) toArray
    val expr = tensorFromArray(inputArray, Array(4, 4, 4))
      .flatMap(slice(_, Array((1, 2), (1, 2), (1, 2))))
      .flatMap({tensor =>
        tensor.magnitude should be (Array(2, 2, 2))
        tensor.order should be (3)
        tensor.elementSize should be (8)

        tensor.valueAt(Array(0, 0, 0)) should be (21)
        tensor.valueAt(Array(1, 0, 1)) should be (38)
        tensor.valueAt(Array(0, 1, 0)) should be (25)

        unit()
      })

    val interp = IdInterpreter
    val _ = interp.eval(expr)

    succeed
  }

  "Taking a slice outside source magnitude" should "not be allowed" in {
    import IntTensorAlgebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 63) toArray, Array(4, 4, 4))
        .flatMap(slice(_, Array((1, 2), (1, 5), (1, 2))))

      val interp = IdInterpreter
      val _ = interp.eval(expr)

      succeed
    }
  }

  "Specifying (0, 1) range for higher dimensions than source order in a slice" should "not cause a problem" in {
    import IntTensorAlgebra._

    val expr = tensorFromArray((0 to 63) toArray, Array(4, 4, 4))
      .flatMap(slice(_, Array((1, 2), (1, 2), (1, 2), (0, 1), (0, 1), (0, 1))))
      .flatMap({ t =>
        t.order should be (3)
        t.elementSize should be (8)

        unit()
      })

    val interp = IdInterpreter
    val _ = interp.eval(expr)
  }

  "Reshaping a tensor" should "yield a tensor with expected magnitude, roder, elementSize, and element values" in {
    import IntTensorAlgebra._

    val expr = tensorFromArray((0 to 59) toArray, Array(10, 2, 3))
      .flatMap(reshape(_, Array(3, 4, 5)))
      .flatMap({ t =>
        t.magnitude should be (Array(3, 4, 5))
        t.order should be (3)
        t.elementSize should be (60)

        //...choose a few different values to assute reshaped indexes give
        //   expected values...
        t.valueAt(Array(0, 0, 0)) should be (0)
        t.valueAt(Array(2, 2, 1)) should be (20)
        t.valueAt(Array(0, 1, 4)) should be (51)
        t.valueAt(Array(1, 3, 2)) should be (34)
        t.valueAt(Array(2, 3, 4)) should be (59)

        unit()
      })

    val interp = IdInterpreter
    val _ = interp.eval(expr)
  }

  "Reshaping a tensor with illegal magnitude values" should "not be allowed" in {
    import IntTensorAlgebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 63) toArray, Array(4, 4, 4))
        .flatMap(reshape(_, Array(16, 0, 4)))

      val interp = IdInterpreter
      val _ = interp.eval(expr)

      succeed
    }
  }

  "Reshaping a tensor with different elementSize" should "not be allowed" in {
    import IntTensorAlgebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 63) toArray, Array(4, 4, 4))
        .flatMap(reshape(_, Array(5, 5, 3)))

      val interp = IdInterpreter
      val _ = interp.eval(expr)

      succeed
    }
  }

  "Reversing a 3x2x3 tensor in the _Y direction" should "yield a tensor with expected magnitude, roder, elementSize, and element values" in {
    import IntTensorAlgebra._

    val expr = tensorFromArray((0 to 11) toArray, Array(2, 3, 2))
      .flatMap(reverse(_, _Y))
      .flatMap({t =>
        t.magnitude should be (Array(2, 3, 2))
        t.order should be (3)
        t.elementSize should be (12)

        t.valueAt(Array(0, 0, 0)) should be (4)
        t.valueAt(Array(1, 0, 0)) should be (5)
        t.valueAt(Array(0, 1, 0)) should be (2)
        t.valueAt(Array(1, 1, 0)) should be (3)
        t.valueAt(Array(0, 2, 0)) should be (0)
        t.valueAt(Array(1, 2, 0)) should be (1)
        t.valueAt(Array(0, 0, 1)) should be (10)
        t.valueAt(Array(1, 0, 1)) should be (11)
        t.valueAt(Array(0, 1, 1)) should be (8)
        t.valueAt(Array(1, 1, 1)) should be (9)
        t.valueAt(Array(0, 2, 1)) should be (6)
        t.valueAt(Array(1, 2, 1)) should be (7)

        unit()
      })

    val interp = IdInterpreter
    val _ = interp.eval(expr)

    succeed
  }

  "Using a negative dimension value when trying to construct a revered tensor" should "not be allowed" in {
    import IntTensorAlgebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 11) toArray, Array(2, 3, 2))
        .flatMap(reverse(_, -1.asInstanceOf[Dimension]))

      val interp = IdInterpreter
      val _ = interp.eval(expr)

      succeed
    }
  }
}
