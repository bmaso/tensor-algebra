package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import cats.implicits._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

/**
 * This test assures that the JVM Int evaluator correctly implements the
 * common abstract operations by putting them through their paces:
 * * tensorFromArray
 * * copyTensorElementsToArray
 **/
class IntArrayEvalSpec extends FlatSpec {
  "Copying array to array" should "transfer array contents unchanged" in {
    val magnitude = Array(5, 5)
    val inputArray = (0 to 24) toArray
    val outputArray = Array.fill[Int](25)(0)

    val expr = IntTensorAlgebra.tensorFromArray(inputArray, Array(5, 5))
        .flatMap(tensor => IntTensorAlgebra.copyTensorElementsToArray(tensor, outputArray))

    val interp = IdInterpreter

    val _ = interp.eval(expr)

    outputArray should be (inputArray)
  }
}
