package bmaso.tensoralg.jvm.integer

import scala.reflect.ClassTag

import org.scalatest.FlatSpec

/**
 * This is a convenient base class for all of the specs testing individual Tensor
 * implementations to extend. I injects `.asI` and .`asTArray` extended functions,
 * making it very easy to create several different versions of the same test, each
 * parameterized by `Byte`, `Short`, `Int`, `Long`, `Float` and
 * `Double` tesnor element values. See, for example, `BroadcastTensorSpecBase and
 * the various subtypes.
 **/
abstract class TensorFlatSpecBase[T: Numeric: ClassTag: ConvertsInt] extends FlatSpec {
  val numeric: Numeric[T] = implicitly[Numeric[T]]

  implicit def canConvertToInt(t: T) = new {
    def asI: Int = numeric.toInt(t)
  }

  implicit def canConvertFromInt(i: Int) = new {
    def asT: T = implicitly[ConvertsInt[T]].convertInt(i)
  }

  implicit def canConvertIntArray(array: Array[Int]) = new {
    def asTArray: Array[T] = implicitly[ConvertsInt[T]].convertIntArray(array)
  }
}
