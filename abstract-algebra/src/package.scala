package bmaso.tensoralg

import scala.language.postfixOps

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
}
