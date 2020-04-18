package bmaso.tensoralg

import cats.free.Free
import scala.concurrent.Future

package object abstractions {

  sealed trait TensorExprAlg[ElT <: TensorElT, T]
  case class TensorFromArray[N, ElT <: TensorElT](arr: Array[N], offset: Int, len: Int)(implicit ev: ScalaElTMatches[N, ElT]) extends TensorExprAlg[ElT, Tensor[ElT]]
  case class CopyTensorElementsToArray[N, ElT <: TensorElT](tensor: Tensor[ElT], arr: Array[N], offset: Int)(implicit ev: ScalaElTMatches[N, ElT]) extends TensorExprAlg[ElT, Either[ComputationError, Tensor[ElT]]]

  type TensorExpr[ElT <: TensorElT, Eff] = Free[({type λ[A] = TensorExprAlg[ElT, A] })#λ, Eff]

  trait ComputationError

  sealed abstract class ComputationComplete
  case object ComputationComplete extends ComputationComplete {
    def computationComplete(): ComputationComplete = this
  }

  /** Tensor element types are restricted to just a few numeric types. All
   * runtime evaluators will support these element types
   * * byte, short, int, long
   * * float, double
   **/
  sealed trait TensorElT
  abstract class ByteElT extends TensorElT
  abstract class ShortElT extends TensorElT
  abstract class IntElT extends TensorElT
  abstract class LongElT extends TensorElT
  abstract class FloatElT extends TensorElT
  abstract class DoubleElT extends TensorElT

  /**
   * Static implicits allowing the compiler to deduce the equivalent Scala
   * numeric types to available Tensor types.
   **/
  sealed trait ScalaElTMatches[N, TensorElT]

//  implicit case object ByteElT extends ByteElT with ScalaElTMatches[Byte, ByteElT]

//  implicit case object ShortElT extends ShortElT with ScalaElTMatches[Short, ShortElT]

//  implicit case object IntElT extends IntElT with ScalaElTMatches[Int, IntElT]

//  implicit case object LongElT extends LongElT with ScalaElTMatches[Long, LongElT]

  /** Declaration of universal support for tensors with `Float` values **/
  implicit case object FloatElT extends FloatElT with ScalaElTMatches[Float, FloatElT]

//  implicit case object DoubleElT extends DoubleElT with ScalaElTMatches[Double, DoubleElT]
}
