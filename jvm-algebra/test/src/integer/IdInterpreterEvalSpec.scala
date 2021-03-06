package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

import bmaso.tensoralg.abstractions.{TensorAlgebra => abstract_TensorAlgebra, _}

/**
 * This test assures that the JVM Int evaluator correctly implements the
 * common abstract operations by putting them through their paces:
 * * tensorFromArray
 * * copyTensorElementsToArray
 **/
class IdInterpreterEvalSpecBase[T: Numeric: ClassTag: ConvertsInt](algebra: TensorAlgebra[T]) extends TensorFlatSpecBase[T] {
  "An eval of Copying array to array" should "transfer array contents unchanged" in {
    import algebra._

    val magnitude = Array(5, 5)
    val inputArray = (0 to 24).toArray.asTArray
    val outputArray = Array.fill[Int](25)(0).asTArray

    val expr = tensorFromArray(inputArray, Array(5, 5))
        .flatMap(tensor => copyTensorElementsToArray(tensor, outputArray))

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    outputArray should be (inputArray)
  }

  "An eval of translating 3x3 array (1, 1)" should "yield shifted elements backfilled with 0s" in {
    import algebra._

    val inputArray = (1 to 9).toArray.asTArray
    val outputArray = Array.fill[Int](9)(0).asTArray

    val expr = tensorFromArray(inputArray, Array(3, 3))
      .flatMap(translate(_, Array(1, 1)))
      .flatMap(copyTensorElementsToArray(_, outputArray))

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    outputArray should be (Array(0, 0, 0,
                                 0, 1, 2,
                                 0, 4, 5).asTArray)
  }

  "An eval of broadcasting a 4x2 to 4x2x2" should "yield a tensor with magnitude, order, elementSize, and element values as expected" in {
    import algebra._

    val expr = tensorFromArray((0 to 7).toArray.asTArray, Array(4, 2))
      .flatMap(broadcast(_, _Z, 2))
      .flatMap({t =>
        t.magnitude should be (Array(4, 2, 2))
        t.order should be (3)
        t.elementSize should be (16)

        for(idx <- 0 to 7) {
          t.valueAt1D(idx) should be (t.valueAt1D(idx + 8))
        }

        unit()
      })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Eval of slicing a 4x4x4 tensor" should "produce tensor with expected magnitude, order, elementSize and element values" in {
    import algebra._

    val inputArray = (0 to 63).toArray.asTArray
    val expr = tensorFromArray(inputArray, Array(4, 4, 4))
      .flatMap(slice(_, Array((1, 2), (1, 2), (1, 2))))
      .flatMap({tensor =>
        tensor.magnitude should be (Array(2, 2, 2))
        tensor.order should be (3)
        tensor.elementSize should be (8)

        tensor.valueAt(Array(0, 0, 0)).asI should be (21)
        tensor.valueAt(Array(1, 0, 1)).asI should be (38)
        tensor.valueAt(Array(0, 1, 0)).asI should be (25)

        unit()
      })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Taking a slice outside source magnitude" should "not be allowed" in {
    import algebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 63).toArray.asTArray, Array(4, 4, 4))
        .flatMap(slice(_, Array((1, 2), (1, 5), (1, 2))))

      val interp = IdInterpreterBuilder.interpreterOf(algebra)
      val _ = expr.foldMap(interp)

      succeed
    }
  }

  "Specifying (0, 1) range for higher dimensions than source order in a slice" should "not cause a problem" in {
    import algebra._

    val expr = tensorFromArray((0 to 63).toArray.asTArray, Array(4, 4, 4))
      .flatMap(slice(_, Array((1, 2), (1, 2), (1, 2), (0, 1), (0, 1), (0, 1))))
      .flatMap({ t =>
        t.order should be (3)
        t.elementSize should be (8)

        unit()
      })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)
  }

  "Reshaping a tensor" should "yield a tensor with expected magnitude, order, elementSize, and element values" in {
    import algebra._

    val expr = tensorFromArray((0 to 59).toArray.asTArray, Array(10, 2, 3))
      .flatMap(reshape(_, Array(3, 4, 5)))
      .flatMap({ t =>
        t.magnitude should be (Array(3, 4, 5))
        t.order should be (3)
        t.elementSize should be (60)

        //...choose a few different values to assute reshaped indexes give
        //   expected values...
        t.valueAt(Array(0, 0, 0)).asI should be (0)
        t.valueAt(Array(2, 2, 1)).asI should be (20)
        t.valueAt(Array(0, 1, 4)).asI should be (51)
        t.valueAt(Array(1, 3, 2)).asI should be (34)
        t.valueAt(Array(2, 3, 4)).asI should be (59)

        unit()
      })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Reshaping a tensor with illegal magnitude values" should "not be allowed" in {
    import algebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 63).toArray.asTArray, Array(4, 4, 4))
        .flatMap(reshape(_, Array(16, 0, 4)))

      val interp = IdInterpreterBuilder.interpreterOf(algebra)
      val _ = expr.foldMap(interp)

      succeed
    }
  }

  "Reshaping a tensor with different elementSize" should "not be allowed" in {
    import algebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 63).toArray.asTArray, Array(4, 4, 4))
        .flatMap(reshape(_, Array(5, 5, 3)))

      val interp = IdInterpreterBuilder.interpreterOf(algebra)
      val _ = expr.foldMap(interp)

      succeed
    }
  }

  "Reversing a 3x2x3 tensor in the _Y direction" should "yield a tensor with expected magnitude, order, elementSize, and element values" in {
    import algebra._

    val expr = tensorFromArray((0 to 11).toArray.asTArray, Array(2, 3, 2))
      .flatMap(reverse(_, _Y))
      .flatMap({t =>
        t.magnitude should be (Array(2, 3, 2))
        t.order should be (3)
        t.elementSize should be (12)

        t.valueAt(Array(0, 0, 0)).asI should be (4)
        t.valueAt(Array(1, 0, 0)).asI should be (5)
        t.valueAt(Array(0, 1, 0)).asI should be (2)
        t.valueAt(Array(1, 1, 0)).asI should be (3)
        t.valueAt(Array(0, 2, 0)).asI should be (0)
        t.valueAt(Array(1, 2, 0)).asI should be (1)
        t.valueAt(Array(0, 0, 1)).asI should be (10)
        t.valueAt(Array(1, 0, 1)).asI should be (11)
        t.valueAt(Array(0, 1, 1)).asI should be (8)
        t.valueAt(Array(1, 1, 1)).asI should be (9)
        t.valueAt(Array(0, 2, 1)).asI should be (6)
        t.valueAt(Array(1, 2, 1)).asI should be (7)

        unit()
      })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Using a negative dimension value when trying to construct a revered tensor" should "not be allowed" in {
    import algebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 11).toArray.asTArray, Array(2, 3, 2))
        .flatMap(reverse(_, -1.asInstanceOf[Dimension]))

      val interp = IdInterpreterBuilder.interpreterOf(algebra)
      val _ = expr.foldMap(interp)

      succeed
    }
  }

  "An (_X, _Y) pivot of a 3x2 tensor" should "yield a tensor with expected magnitude, order, elementSize, and element values" in {
    import algebra._

    val expr = tensorFromArray((0 to 5).toArray.asTArray, Array(3, 2))
      .flatMap(pivot(_, _X, _Y))
      .flatMap({t =>
        t.magnitude should be (Array(2, 3))
        t.order should be (2)
        t.elementSize should be (6)

        t.valueAt(Array(0, 0)).asI should be (0)
        t.valueAt(Array(1, 0)).asI should be (3)
        t.valueAt(Array(0, 1)).asI should be (1)
        t.valueAt(Array(1, 1)).asI should be (4)
        t.valueAt(Array(0, 2)).asI should be (2)
        t.valueAt(Array(1, 2)).asI should be (5)

        unit()
      })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Using an illegal dimension value when constructing a pivot" should "not be allowed" in {
    import algebra._

    intercept[IllegalArgumentException] {
      val expr = tensorFromArray((0 to 5).toArray.asTArray, Array(3, 2, 2))
        .flatMap(pivot(_, -1.asInstanceOf[Dimension], -1.asInstanceOf[Dimension]))

      val interp = IdInterpreterBuilder.interpreterOf(algebra)
      val _ = expr.foldMap(interp)

      succeed
    }
  }

  "Joining two 3x2 tensors in the _Z dimension" should "yield a StackTensor with expected magnitude, order, elementSize, and element values" in {
    import algebra._

    val expr = tensorFromArray((0 to 5).toArray.asTArray, Array(3, 2))
      .flatMap(t1 =>
        tensorFromArray((6 to 11).toArray.asTArray, Array(3, 2))
          .flatMap(t2 => join(_Z, t1, t2)))
      .flatMap({t =>
        t.isInstanceOf[StackTensor[T]] should be (true)

        t.magnitude should be (Array(3, 2, 2))
        t.order should be (3)
        t.elementSize should be (12)

        t.valueAt(Array(0, 0, 0)).asI should be (0)
        t.valueAt(Array(1, 0, 0)).asI should be (1)
        t.valueAt(Array(2, 0, 0)).asI should be (2)
        t.valueAt(Array(0, 1, 0)).asI should be (3)
        t.valueAt(Array(1, 1, 0)).asI should be (4)
        t.valueAt(Array(2, 1, 0)).asI should be (5)
        t.valueAt(Array(0, 0, 1)).asI should be (6)
        t.valueAt(Array(1, 0, 1)).asI should be (7)
        t.valueAt(Array(2, 0, 1)).asI should be (8)
        t.valueAt(Array(0, 1, 1)).asI should be (9)
        t.valueAt(Array(1, 1, 1)).asI should be (10)
        t.valueAt(Array(2, 1, 1)).asI should be (11)

        unit()
      })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Joining two 3x2 tensors in the _Y dimension" should "yield a JoinTensor with expected magnitude, order, elementSize, and element values" in {
    import algebra._

    val expr = tensorFromArray((0 to 5).toArray.asTArray, Array(3, 2))
      .flatMap(t1 =>
        tensorFromArray((6 to 11).toArray.asTArray, Array(3, 2))
          .flatMap(t2 => join(_Y, t1, t2)))
      .flatMap({t =>
        t.isInstanceOf[JoinTensor[T]] should be (true)

        t.magnitude should be (Array(3, 4))
        t.order should be (2)
        t.elementSize should be (12)

        t.valueAt(Array(0, 0)).asI should be (0)
        t.valueAt(Array(1, 0)).asI should be (1)
        t.valueAt(Array(2, 0)).asI should be (2)
        t.valueAt(Array(0, 1)).asI should be (3)
        t.valueAt(Array(1, 1)).asI should be (4)
        t.valueAt(Array(2, 1)).asI should be (5)
        t.valueAt(Array(0, 2)).asI should be (6)
        t.valueAt(Array(1, 2)).asI should be (7)
        t.valueAt(Array(2, 2)).asI should be (8)
        t.valueAt(Array(0, 3)).asI should be (9)
        t.valueAt(Array(1, 3)).asI should be (10)
        t.valueAt(Array(2, 3)).asI should be (11)

        unit()
      })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Mapping a 2x3 tensors w/ mapping function (_ * 2)" should "yield a tensor with expected magnitude, order, elementSize, and element values" in {
    import algebra._

    val expr = tensorFromArray((0 to 5).toArray.asTArray, Array(2, 3))
        .flatMap(map(_)(el => implicitly[Numeric[T]].times(el, 2.asT)))
        .flatMap({ t =>
          t.magnitude should be (Array(2, 3))
          t.order should be (2)
          t.elementSize should be (6)

          t.valueAt(Array(0, 0)).asI should be (0)
          t.valueAt(Array(1, 0)).asI should be (2)
          t.valueAt(Array(0, 1)).asI should be (4)
          t.valueAt(Array(1, 1)).asI should be (6)
          t.valueAt(Array(0, 2)).asI should be (8)
          t.valueAt(Array(1, 2)).asI should be (10)

          unit()
        })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Reducing a 2x3x4x3 tensor in 2 orders" should "yield a tensor with expected magnitude, order, elementSize, and element values" in {
    import algebra._

    def sum(t: JVMTensor[T]): T = (0 to (t.elementSize.toInt - 1)).map(t.valueAt1D(_)).sum

    val expr = tensorFromArray((0 to 71).map(_ => 1).toArray.asTArray, Array(2, 3, 4, 3))
        .flatMap(reduce(_, 2)(sum))
        .flatMap({ t =>
          t.magnitude should be (Array(4, 3))
          t.order should be (2)
          t.elementSize should be (12)

          t.valueAt(Array(0, 0)).asI should be (6)
          t.valueAt(Array(1, 0)).asI should be (6)
          t.valueAt(Array(0, 1)).asI should be (6)
          t.valueAt(Array(1, 1)).asI should be (6)
          t.valueAt(Array(0, 2)).asI should be (6)
          t.valueAt(Array(1, 2)).asI should be (6)

          unit()
        })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Inverting a scalar" should "basically yield the exact same scalar" in {
    import algebra._

    val expr = tensorFromArray(Array(42).asTArray, Array(1))
        .flatMap(invert(_))
        .flatMap({ t =>
          t.magnitude should be (Array(1))
          t.order should be (1)
          t.elementSize should be (1)

          t.valueAt(Array(0)).asI should be (42)

          unit()
        })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "Inverting a 3x2 tensor" should "yield a 2x3 tensor with expected magnitude, order, elementSize and element values" in {
    import algebra._

    val expr = tensorFromArray((0 to 5).toArray.asTArray, Array(3, 2))
        .flatMap(invert(_))
        .flatMap({ t =>
          t.magnitude should be (Array(2, 3))
          t.order should be (2)
          t.elementSize should be (6)

          t.valueAt(Array(0, 0)).asI should be (0)
          t.valueAt(Array(1, 0)).asI should be (3)
          t.valueAt(Array(0, 1)).asI should be (1)
          t.valueAt(Array(1, 1)).asI should be (4)
          t.valueAt(Array(0, 2)).asI should be (2)
          t.valueAt(Array(1, 2)).asI should be (5)

          unit()
        })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }


  "Inverting a 4x3x2 tensor" should "yield a 2x3x4 tensor with expected magnitude, order, elementSize and element values" in {
    import algebra._

    val expr = tensorFromArray((0 to 23).toArray.asTArray, Array(4, 3, 2))
        .flatMap(invert(_))
        .flatMap({ t =>
          t.magnitude should be (Array(2, 3, 4))
          t.order should be (3)
          t.elementSize should be (24)

          t.valueAt(Array(0, 0, 0)).asI should be (0)
          t.valueAt(Array(1, 0, 0)).asI should be (12)
          t.valueAt(Array(0, 1, 0)).asI should be (4)
          t.valueAt(Array(1, 1, 0)).asI should be (16)
          t.valueAt(Array(0, 2, 0)).asI should be (8)
          t.valueAt(Array(1, 2, 0)).asI should be (20)
          t.valueAt(Array(0, 0, 1)).asI should be (1)
          t.valueAt(Array(1, 0, 1)).asI should be (13)
          t.valueAt(Array(0, 1, 1)).asI should be (5)
          t.valueAt(Array(0, 2, 1)).asI should be (9)
          t.valueAt(Array(1, 1, 1)).asI should be (17)
          t.valueAt(Array(1, 2, 1)).asI should be (21)
          t.valueAt(Array(0, 0, 2)).asI should be (2)
          t.valueAt(Array(1, 0, 2)).asI should be (14)
          t.valueAt(Array(0, 1, 2)).asI should be (6)
          t.valueAt(Array(1, 1, 2)).asI should be (18)
          t.valueAt(Array(0, 2, 2)).asI should be (10)
          t.valueAt(Array(1, 2, 2)).asI should be (22)
          t.valueAt(Array(0, 0, 3)).asI should be (3)
          t.valueAt(Array(1, 0, 3)).asI should be (15)
          t.valueAt(Array(0, 1, 3)).asI should be (7)
          t.valueAt(Array(1, 1, 3)).asI should be (19)
          t.valueAt(Array(0, 2, 3)).asI should be (11)
          t.valueAt(Array(1, 2, 3)).asI should be (23)

          unit()
        })

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "matmult2D of a 3x2 tensor and a 2x3 tensor" should "yield a 2x2 tensor of the expected magnitude, order, elementSize and element values" in {
    import algebra._

    val expr =
      for(t1 <- tensorFromArray((0 to 5).toArray.asTArray, Array(3, 2));
          t2 <- tensorFromArray((0 to 5).toArray.asTArray, Array(2, 3));
          mm <- matmult2D(t1, t2)) yield {

        mm.magnitude should be (Array(2, 2))
        mm.order should be (2)
        mm.elementSize should be (4)

        mm.valueAt(Array(0, 0)).asI should be (10)
        mm.valueAt(Array(1, 0)).asI should be (13)
        mm.valueAt(Array(0, 1)).asI should be (28)
        mm.valueAt(Array(1, 1)).asI should be (40)

        unit()
      }

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "matmult2D of a 3x4 tensor and a 4x2 tensor" should "yield a 3x2 tensor of the expected magnitude, order, elementSize and element values" in {
    import algebra._

    val expr =
      for(t1 <- tensorFromArray((0 to 11).toArray.asTArray, Array(4, 3));
          t2 <- tensorFromArray((0 to 7).toArray.asTArray, Array(2, 4));
          mm <- matmult2D(t1, t2)) yield {

        mm.magnitude should be (Array(2, 3))
        mm.order should be (2)
        mm.elementSize should be (6)

        mm.valueAt(Array(0, 0)) should be (28.asT)
        mm.valueAt(Array(1, 0)) should be (34.asT)
        mm.valueAt(Array(0, 1)) should be (76.asT)
        mm.valueAt(Array(1, 1)) should be (98.asT)
        mm.valueAt(Array(0, 2)) should be (124.asT)
        mm.valueAt(Array(1, 2)) should be (162.asT)

        unit()
      }

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "cross-correlation of a 3x3 uniform tensor with a symmetric 3x3 kernel" should "yield a 3x3 tensor of the expected magnitude, order, elementSize and element values" in {
    import algebra._

    val sourceArray = Array(
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
    ).asTArray

    val kernelArray = Array(
      0, 1, 0,
      1, 0, 1,
      0, 1, 0,
    ).asTArray

    val expected = Array(
      2, 3, 2,
      3, 4, 3,
      2, 3, 2,
    ).asTArray

    val resultsArray = Array.fill[Int](9)(0).asTArray

    val expr =
      for(tensor <- tensorFromArray(sourceArray, Array(3, 3));
          kernel <- tensorFromArray(kernelArray, Array(3, 3));
          cc     <- crossCorrelate(tensor, kernel);
          _      <- copyTensorElementsToArray(cc, resultsArray, 0)) yield {
        cc.magnitude should be (Array(3, 3))
        cc.order should be (2)
        cc.elementSize should be (9)

        resultsArray should be (expected)

        unit
      }

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "cross-correlation of a 3x3 uniform tensor with an asymmetric 3x3 kernel" should "yield a 3x3 tensor of the expected magnitude, order, elementSize and element values" in {
    import algebra._

    val sourceArray = Array(
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
    ).asTArray

    val kernelArray = Array(
      0, 1, 2,
      2, 1, 0,
      0, 0, 1,
    ).asTArray

    val expected = Array(
      2, 4, 3,
      5, 7, 4,
      4, 6, 4,
    ).asTArray

    val resultsArray = Array.fill[Int](9)(0).asTArray

    val expr =
      for(tensor <- tensorFromArray(sourceArray, Array(3, 3));
          kernel <- tensorFromArray(kernelArray, Array(3, 3));
          cc     <- crossCorrelate(tensor, kernel);
          _      <- copyTensorElementsToArray(cc, resultsArray, 0)) yield {
        cc.magnitude should be (Array(3, 3))
        cc.order should be (2)
        cc.elementSize should be (9)

        resultsArray should be (expected)

        unit
      }

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }

  "cross-correlation of a 3x3 non-uniform tensor with an asymmetric 3x3 kernel" should "yield a 3x3 tensor of the expected magnitude, order, elementSize and element values" in {
    import algebra._

    val sourceArray = Array(
      0, 1, 2,
      3, 4, 5,
      6, 7, 8,
    ).asTArray

    val kernelArray = Array(
      0, 1, 2,
      2, 1, 0,
      0, 0, 1,
    ).asTArray

    val expected = Array(
      4,  6,  4,
      12, 23, 15,
      17, 33, 27,
    ).asTArray

    val resultsArray = Array.fill[Int](9)(0).asTArray

    val expr =
      for(tensor <- tensorFromArray(sourceArray, Array(3, 3));
          kernel <- tensorFromArray(kernelArray, Array(3, 3));
          cc     <- crossCorrelate(tensor, kernel);
          _      <- copyTensorElementsToArray(cc, resultsArray, 0)) yield {
        cc.magnitude should be (Array(3, 3))
        cc.order should be (2)
        cc.elementSize should be (9)

        resultsArray should be (expected)

        unit
      }

    val interp = IdInterpreterBuilder.interpreterOf(algebra)
    val _ = expr.foldMap(interp)

    succeed
  }
}

class IntIdInterpreterEvalSpec extends IdInterpreterEvalSpecBase[Int](IntTensorAlgebra)
