package bmaso.tensoralg

import scala.language.postfixOps

import cats.free.Free
import scala.concurrent.Future
import com.softwaremill.tagging._

package object abstractions {

  /** A tag type used to define the phanton type `Dimension` */
  trait DimensionTag
  /** `Dimension` is `Int @@ DimensionTag` */
  type Dimension = Int @@ DimensionTag

  val Seq(
      _X : Dimension, _Y : Dimension, _Z : Dimension, _W : Dimension,
      _V : Dimension, _U : Dimension, _T : Dimension, _S : Dimension,
      _R : Dimension, _Q : Dimension, _P : Dimension, _O : Dimension,
      _N : Dimension, _M : Dimension, _L : Dimension, _K : Dimension,
      _J : Dimension, _I : Dimension, _H : Dimension, _G : Dimension,
      _F : Dimension, _E : Dimension) =
    (0 to 21) map (_.asInstanceOf[Dimension]) toSeq

  val Seq(
      _D : Dimension, _C : Dimension, _B : Dimension, _A : Dimension,
      _XX: Dimension, _YY: Dimension, _ZZ: Dimension, _WW: Dimension,
      _VV: Dimension, _UU: Dimension, _TT: Dimension, _SS: Dimension,
      _RR: Dimension, _QQ: Dimension, _PP: Dimension, _OO: Dimension,
      _NN: Dimension, _MM: Dimension, _LL: Dimension, _KK: Dimension,
      _JJ: Dimension, _II: Dimension) =
    (22 to 43) map (_.asInstanceOf[Dimension]) toSeq

  val Seq(
      _HH: Dimension, _GG: Dimension, _FF: Dimension, _EE: Dimension,
      _DD: Dimension, _CC: Dimension, _BB: Dimension, _AA: Dimension) =
  (44 to 51) map (_.asInstanceOf[Dimension]) toSeq

  val NaturalDimensionSequencing = (0 to 51) map (_.asInstanceOf[Dimension]) toArray
  val UnitaryStepping: Array[Long] = (0 to 100) map (_ => 1L) toArray
  val MaximalGrouping: Array[Long] = (0 to 100) map (_ => Long.MaxValue) toArray

  /**
   * The tensor algebraic expressions which are common to all evaluators. Using the
   * *Free monad* pattern: allows me to compose evaluator-specific algebras
   * in order to specialize ***reduce*** operation definitions for different
   * evaluators.
   **/
  sealed trait TensorExprAlg[ElT <: TensorElT, T]
  case class TensorFromArray[N, ElT <: TensorElT](arr: Array[N], offset: Int, len: Int)(implicit ev: ScalaElTMatches[N, ElT]) extends TensorExprAlg[ElT, Tensor[ElT]]
  case class CopyTensorElementsToArray[N, ElT <: TensorElT](tensor: Tensor[ElT], arr: Array[N], offset: Int)(implicit ev: ScalaElTMatches[N, ElT]) extends TensorExprAlg[ElT, Either[ComputationError, Tensor[ElT]]]
  case class Translate[ElT <: TensorElT](tensor: Tensor[ElT], dimensions: Array[Dimension], offsets: Array[Long]) extends TensorExprAlg[ElT, Tensor[ElT]]
  case class Broadcast[ElT <: TensorElT](tensor: Tensor[ElT], baseMagnitude: Array[Long]) extends TensorExprAlg[ElT, Tensor[ElT]]
  case class Reshape[ElT <: TensorElT](tensor: Tensor[ElT], reshapedMagnitude: Array[Long], sourceDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing, targetDimensionSequencing: Array[Dimension] = NaturalDimensionSequencing) extends TensorExprAlg[ElT, Tensor[ElT]]
  case class Split[ElT <: TensorElT](tensor: Tensor[ElT], splitDimensions: Array[Dimension], splitStepping: Array[Long] = UnitaryStepping) extends TensorExprAlg[ElT, Array[Tensor[ElT]]]
  case class Join[ElT <: TensorElT](tensors: Array[Tensor[ElT]], joiningDimensions: Array[Dimension], joinGrouping: Array[Long] = MaximalGrouping) extends TensorExprAlg[ElT, Tensor[ElT]]

  /*
   * Coding note: A Reduce expression requires evaluator-specific representation.
   * For example, JVM class in the JVM evaluator, Aparapi-constrained class in the
   * Aparapi evaluator, or OpenCL-coding in the OpenCL evaluator. Other evaluators
   * dreamed up in the future will have their own reduction representations.
   *
   * There will be A separate algebra for each evaluator defining evaluator-specific
   * operations, such as the evaluator-specific Reduce. Then, because the Free
   * monad design makes it possible, we will combine the generic TensorExprAlg
   * with evaluator-specific algebra(s) to create combo algebras. The various
   * Free monad interpreters/compilers are based on these combo algebras.
   */

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
