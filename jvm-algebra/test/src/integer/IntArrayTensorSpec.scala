package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

class IntArrayTensorSpec extends FlatSpec {
  "A 5x5 tensor" should "have expected magnitude, order, and elementSize" in {
    val t = IntArrayTensor((0 to 24) toArray, Array(5, 5), 0, 25)
    t.order should be (2)
    t.magnitude(_X) should be (5)
    t.magnitude(_Y) should be (5)
    t.elementSize should be (25)
  }

  "An attempt to create a tensor with magnitude/length mismatch" should "throw IllegalArgumentException" in {
    intercept[IllegalArgumentException] {
      IntArrayTensor((1 to 100) toArray, Array(5, 5), 0, 100)
    }
  }
}
