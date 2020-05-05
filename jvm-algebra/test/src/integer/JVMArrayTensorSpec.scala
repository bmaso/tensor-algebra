package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps
import scala.reflect.ClassTag

import bmaso.tensoralg.abstractions._
import org.scalatest.{FlatSpec, Matchers}, Matchers._

abstract class JVMArrayTensorSpecBase[T: Numeric: ClassTag: ConvertsIntArray] extends FlatSpec {
  val numeric: Numeric[T] = implicitly[Numeric[T]]

  implicit def canConvertToInt(t: T) = new {
    def asI: Int = numeric.toInt(t)
  }

  implicit def canConvertIntArray(array: Array[Int]) = new {
    def asTArray: Array[T] = {
      implicitly[ConvertsIntArray[T]].convertIntArray(array)
    }
  }

  "A 5x5 tensor" should "have expected magnitude, order, and elementSize" in {
    val t = ArrayTensor[T](((0 to 24) toArray).asTArray, Array(5, 5), 0)
    t.order should be (2)
    t.magnitude(_X) should be (5)
    t.magnitude(_Y) should be (5)
    t.elementSize should be (25)
  }

  it should "yield expected values from valueAt" in {
    val t = ArrayTensor[T](((0 to 24) toArray).asTArray, Array(5, 5), 0)
    t.valueAt(Array(0, 0)).asI should be (0)
    t.valueAt(Array(2, 0)).asI should be (2)
    t.valueAt(Array(3, 3)).asI should be (18)
    t.valueAt(Array(3, 1)).asI should be (8)
    t.valueAt(Array(4, 4)).asI should be (24)
  }

  it should "yield expected values from valueAt1D" in {
    val t = ArrayTensor[T](((0 to 24) toArray).asTArray, Array(5, 5), 0)
    t.valueAt1D(0).asI should be (0)
    t.valueAt1D(10).asI should be (10)
    t.valueAt1D(18).asI should be (18)
    t.valueAt1D(16).asI should be (16)
    t.valueAt1D(24).asI should be (24)
  }

  "An attempt to create a tensor with magnitude/length mismatch" should "throw IllegalArgumentException" in {
    intercept[IllegalArgumentException] {
      ArrayTensor[T](((1 to 100) toArray).asTArray, Array(5, 25), 0)
    }
  }

  "a tensor" should "ignore additional unitary values in index passed to valueAt" in {
    val t = ArrayTensor[T](((0 to 63) toArray).asTArray, Array(4, 4, 4), 0)

    t.valueAt(Array(0, 0, 0)).asI should be (0)
    t.valueAt(Array(0, 0, 0, 0, 0, 0)).asI should be (0)

    t.valueAt(Array(1, 1, 1)).asI should be (21)
    t.valueAt(Array(1, 1, 1, 0, 0, 0)).asI should be (21)
  }

  "Accessing an element with an index out of range" should "not be allowed" in {
    val t = ArrayTensor[T](((0 to 63) toArray).asTArray, Array(4, 4, 4), 0)
    intercept[IllegalArgumentException] {
      t.valueAt(Array(0, 5, 0))
    }
  }

  "Accessing an element with an index out of range in undeclared dimension" should "not be allowed" in {
    val t = ArrayTensor[T](((0 to 63) toArray).asTArray, Array(4, 4, 4), 0)
    intercept[IllegalArgumentException] {
      t.valueAt(Array(0, 0, 0, 1))
    }
  }
}

trait JVMArrayLargerTypesSpecTrait[T] {
    this: JVMArrayTensorSpecBase[T] =>

  "A 7x3x8x2 tensor" should "have expected values from valueAt" in {
    val t = ArrayTensor[T](((0 to 335) toArray).asTArray, Array(7, 3, 8, 2), 0)
    t.valueAt(Array(0, 0, 0, 0)).asI should be (0)
    t.valueAt(Array(2, 0, 0, 0)).asI should be (2)
    t.valueAt(Array(4, 2, 5, 1)).asI should be (291.asInstanceOf[T]) //  4 + 14 + 105 + 168
    t.valueAt(Array(6, 2, 7, 1)).asI should be (335.asInstanceOf[T]) //  6 + 14 + 147 + 168
  }

  it should "yield expected values from valueAt1D" in {
    val t = ArrayTensor[T](((0 to 335) toArray).asTArray, Array(7, 3, 8, 2), 0)
    t.valueAt1D(0).asI should be (0)
    t.valueAt1D(96).asI should be (96)
    t.valueAt1D(235).asI should be (235)
    t.valueAt1D(335).asI should be (335)
  }
}

class ByteJVMArrayTensorSpec extends JVMArrayTensorSpecBase[Byte] {
  "A 4x3x3x2 tensor" should "have expected values from valueAt" in {
    val t = ArrayTensor[Byte](((0 to 71) toArray).asTArray, Array(4, 3, 3, 2), 0)
    t.valueAt(Array(0, 0, 0, 0)) should be (0: Byte)
    t.valueAt(Array(2, 0, 0, 0)) should be (2: Byte)
    t.valueAt(Array(3, 2, 1, 0)) should be (23: Byte)
    t.valueAt(Array(3, 2, 2, 1)) should be (71: Byte)
  }

  it should "yield expected values from valueAt1D" in {
    val t = ArrayTensor[Byte](((0 to 71) toArray).asTArray, Array(4, 3, 3, 2), 0)
    t.valueAt1D(0) should be (0: Byte)
    t.valueAt1D(2) should be (2: Byte)
    t.valueAt1D(23) should be (23: Byte)
    t.valueAt1D(71) should be (71: Byte)
  }
}

class ShortJVMArrayTensorSpec extends JVMArrayTensorSpecBase[Short] with JVMArrayLargerTypesSpecTrait[Short]
class IntJVMArrayTensorSpec extends JVMArrayTensorSpecBase[Int] with JVMArrayLargerTypesSpecTrait[Int]
class LongJVMArrayTensorSpec extends JVMArrayTensorSpecBase[Long] with JVMArrayLargerTypesSpecTrait[Long]
class FloatJVMArrayTensorSpec extends JVMArrayTensorSpecBase[Float] with JVMArrayLargerTypesSpecTrait[Float]
class DoubleJVMArrayTensorSpec extends JVMArrayTensorSpecBase[Double] with JVMArrayLargerTypesSpecTrait[Double]
