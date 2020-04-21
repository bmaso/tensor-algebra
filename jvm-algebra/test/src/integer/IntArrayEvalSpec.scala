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

  ignore /*"An eval of ranslating 3x3 array (1, 1)"*/ should "yield shifted elements backfilled with 0s" in {
    import IntTensorAlgebra._

    val inputArray = (1 to 9) toArray
    val outputArray = Array.fill[Int](9)(0)

    val expr = tensorFromArray(inputArray, Array(3, 3))
      .flatMap(translate(_, Array(_X, _Y), Array(1, 1)))
      .flatMap(copyTensorElementsToArray(_, outputArray))

    val interp = IdInterpreter
    val _ = interp.eval(expr)

    outputArray should be (Array(0, 0, 0,
                                 0, 1, 2,
                                 0, 4, 5))
  }
}
