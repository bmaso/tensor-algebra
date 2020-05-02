package bmaso.tensoralg.abstractions

import scala.language.postfixOps

import cats.free.Free

import scala.annotation.tailrec
import scala.math.{max, min}

/**
 * Each evaluator defines its own concrete tensor type, and defines its operations
 * in terms of this tensor type.
 **/
trait TensorAlgebra {
  type Tensor <: bmaso.tensoralg.abstractions.Tensor
  // type AggregatingTensor <: this.Tensor with Aggregating[this.Tensor]
  // trait AggregatingTensorBuilder {
  //   def freshTensor(magnitude: Array[Long]): TensorAlgebra.this.AggregatingTensor
  // }

  /**
   * Ambient object able to create a "fresh" aggregating tensor. The aggregating
   * tensor should have a "zero" value, meaning that after it aggregates a tensor
   * ***A***, then the element values for the resultant aggregate should be identical
   * to the element values of ***A***.
   **/
  // val aggregatingTensorBuilder: AggregatingTensorBuilder

  type MapFunction
  type ReduceFunction

  /**
   * To define such operations as matrix multiplication and cross-correlation
   * for all algebras, each algebra must have a SUM reduction funtion.
   **/
  val SUM: ReduceFunction

  /**
   * To define such operations as matrix multiplication and cross-correlation
   * for all algebras, each algebra must have a PRODUCT reduction funtion.
   **/
  val PRODUCT: ReduceFunction

  // type TransformFunction

  /**
   * The tensor algebraic expressions which are common to all concrete tensor algebras.
   * Note what additional expression types must be added to concrete subtypes:
   * * Expressions for constructing tensors from Scala data
   * * Expressions for retrieving data elements from tensors
   * * ***reduce* expressions
   **/
  trait TensorExprOp[T]

  /**
   * Represents the translation of a tensor in zero or more dimensions.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `translate` to perform invariant checks.
   **/
  case class Translate(tensor: this.Tensor, offsets: Array[Long]) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the expansion of a tensor in one dimension by duplication.
   **/
  case class Broadcast(tensor: this.Tensor, dimension: Dimension, magnitude: Long) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the re-indexing of a tensor: this changes the dimensionality without
   * changing the `elementSize`.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `reshape` to perform invariant checks.
   **/
  case class Reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long]) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents a segment of a tensor as a tensor.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `slice` to perform invariant checks.
   **/
  case class Slice(tensor: this.Tensor, sliceRange: Array[(Long, Long)]) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents construction of a tensor from two or more base tensors.
   *
   * As with all `TensorExprOp`, invarant assertions are not garuanteed to be performed.
   * Use function `join` to perform invariant checks.
   **/
  case class Join(tensors: List[this.Tensor], joiningDimension: Dimension) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor by 1:1 mapping of element values.
   *
   * @returns The resulting tensor has the exact same magnitude as the input
   *     tensor.
   **/
  case class Map(tensor: this.Tensor, map_f: this.MapFunction) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor by reducing a source tensor in zero
   * or more dimensions using a reducing function.
   *
   * @returns The resulting tensor is of order `tensor.order - reducedOrders`. The
   *     magnitude or the resulting tensor will be equal to `tensor.magnitude.take(tensor.order - reducedOrders)`
   * @param reduce_f The reducing function must accept tensors of magnitude
   *     `tensor.magnitude.drop(tensor.order - reducedOrders)`
   **/
  case class Reduce(tensor: this.Tensor, reducedOrders: Int, reduce_f: this.ReduceFunction) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor by map-reduce. Conceptually, individual
   * slices of the source tensor are transformed via a transforming function,
   * and the results are aggregated together into an aggregating tensor.
   *
   * @returns The resulting tensor is of magnitude `producedMagnitude`
   *
   * @param transform_f The transforming function must accept tensors of magnitude
   *     `tensor.magnitude.drop(tensor.order - reducedOrders)`, and produce tensors
   *     of magnitude `producedMagnitude`
   * @param builder The tensor builder must generate "zero-valued" `AggregateTensor`
   *     instances of magnitude `producedMagnitude`
   **/
  // case class Aggregate(tensor: this.Tensor, reducedOrders: Int, producedMagnitude: Array[Long], transform_f: this.TransformFunction) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of a tensor from another by reversing index values
   * along one dimension.
   **/
  case class Reverse(tensor: this.Tensor, dimension: Dimension) extends this.TensorExprOp[this.Tensor]

  /**
   * Represents the construction of one tensor from another by exchanging the
   * indexes in exactly 2 dimensions.
   **/
  case class Pivot(tensor: this.Tensor, dim1: Dimension, dim2: Dimension) extends this.TensorExprOp[this.Tensor]

  /**
   * Each algebra must define its own definition of the MorphFunction type. The Algebra should
   * also include a mechanism to construct or obtain instances. Algebras will also have to
   * provide some specific instances: `_SUM`, `_PRODUCT`
   **/
  // type MorphFunction <: bmaso.tensoralg.abstractions.MorphFunction

  /**
   * Represents the nullary operation.
   **/
  case object Unit extends this.TensorExprOp[Unit]

  type TensorExpr[Eff] = Free[TensorExprOp, Eff]

  /**
   * Construct a tensor with the same magnitude as the original, but with values
   * tanslated in zero or more dimensions. Element values translated to indexes
   * outside original magnitude will be truncated. Resultant tensor will be "backfilled"
   * with zero values for elements that don't have a corresponding value in original
   * tensor.
   **/
  def translate(tensor: this.Tensor, offsets: Array[Long]): this.TensorExpr[this.Tensor] =
    //...drop trailing zero offsets for efficiency...
    Free.liftF(Translate(tensor, offsets.reverse.dropWhile(_ == 0).reverse))

  /**
   * Construct a tensor by duplicating an existing tensor in one dimension.
   **/
  def broadcast(tensor: this.Tensor, dimension: Dimension, magnitude: Long): this.TensorExpr[this.Tensor] = {
    val effectiveTensorMag = Array.fill[Long](max(tensor.magnitude.length, dimension + 1))(1L)
    Array.copy(tensor.magnitude, 0, effectiveTensorMag, 0, tensor.magnitude.length)

    if(magnitude % effectiveTensorMag(dimension) != 0)
      throw new IllegalArgumentException("The input magnitude must be a positive multiple of the current magnitude")

    if(magnitude == effectiveTensorMag(dimension))
      Free.pure(tensor)
    else
      Free.liftF(Broadcast(tensor, dimension, magnitude))
  }

  def reshape(tensor: this.Tensor, reshapedMagnitude: Array[Long]): this.TensorExpr[this.Tensor] = {
    if(!reshapedMagnitude.map(_ >= 1).fold(true)(_ && _)) throw new IllegalArgumentException("Reshaped magnitude includes illegal values; all arities must be >= 1")
    if(tensor.elementSize != reshapedMagnitude.fold(1L)(_ * _)) throw new IllegalArgumentException("Reshaped tensor elementSize must equal source elementSize")

    Free.liftF(Reshape(tensor, reshapedMagnitude))
  }

  def slice(tensor: this.Tensor, sliceRange: Array[(Long, Long)]): this.TensorExpr[this.Tensor] = {
    if(!sliceRange.drop(tensor.order).map({ case (f, l) => f == 0 && l == 1}).fold(true)(_ && _)) throw new IllegalArgumentException("Slice range implies higher order than source tensor")
    if(!tensor.magnitude.zip(sliceRange).map({ case (m, (f, l)) => f + l <= m}).fold(true)(_ && _)) throw new IllegalArgumentException("Slice range is outside of source tensor magnitude")
    Free.liftF(Slice(tensor, sliceRange))
  }

  def join(joiningDimension: Dimension, t1: this.Tensor, t2: this.Tensor*): this.TensorExpr[this.Tensor] = {
    val tensors: List[this.Tensor] = List(t1) :++ t2.toList
    Free.liftF(Join(tensors, joiningDimension))
  }

  /**
   * Constructs a tensor by reversing the elements in a specific dimension.
   *
   * "Rotating" a tensor about one or more dimensions can be accomplished by
   * applying different sequences of `reverse` and `pivot`.
   *
   * Note that the same could be accomplished by slicing unitary width tensors in
   * a specific dimension and re-assembling with `join` in opposite order. Defining
   * `reverse` as a basic operation makes this common operation much easier to
   * implement efficiently by initerpreters.
   **/
  def reverse(tensor: this.Tensor, dimension: Dimension): this.TensorExpr[this.Tensor] = {
    if(dimension < 0) throw new IllegalArgumentException("Dimension value is not legal; dimension numbers must be >= 0")
    Free.liftF(Reverse(tensor, dimension))
  }

  /**
   * Pivots a tensor by exchanging 2 dimensions.
   *
   * "Rotating" a tensor about one or more dimensions can be accomplished by
   * applying different sequences of `reverse` and `pivot`.
   **/
  def pivot(tensor: this.Tensor, dim1: Dimension, dim2: Dimension): this.TensorExpr[this.Tensor] = {
    if(dim1 < 0 || dim2 < 0) throw new IllegalArgumentException("One or both dimension value is illegal; dimension numbers must be >= 0")
    Free.liftF(Pivot(tensor, dim1, dim2))
  }

  /**
   * Constructs a new tensor whose elements are a 1:1 mapping of the source
   * tensor.
   * @returns An expression that, when evaluated, yields a tensor with the exact
   *     same magnitude as `tensor.magnitude`
   **/
  def map(tensor: this.Tensor, map_f: this.MapFunction): this.TensorExpr[this.Tensor] =
    Free.liftF(Map(tensor, map_f))

  /**
   * Constructs a new tensor whose elements are a reduction of slices that comprise
   * the source tensor. The slices are defined by iteating over unitary width
   * slices in the first `reduceOrders` dimensions.
   *
   * @returns An expression that, once evaluated, yields a tensor whose magnitude
   *     is `tensor.magnitude.drop(reduceOrders)`
   * @param reduce_f A reduction function that receives slices from the original
   *     source tensor of magnitude `tensor.magnitude.take(reduceOrders)`, and which
   *     yields a unitary value.
   **/
  def reduce(tensor: this.Tensor, reduceOrders: Int, reduce_f: this.ReduceFunction): this.TensorExpr[this.Tensor] = {
    if(reduceOrders < 0) throw new IllegalArgumentException("The reduceOrders value must be >= 0")
    Free.liftF(Reduce(tensor, min(reduceOrders, tensor.order), reduce_f))
  }

  /**
   * Constructs a new tensor built by aggregating transformed slices of the original
   * source tensor.
   *
   * @returns The returned expression, whenn evaluated, yields a tensor of magnitude
   *     `producedMagnitude`
   * @param transform_f A tranform function that receives a tensor of magnitude
   *     `tensor.magnitude.take(reduceOrders)`, and which yields a tensor of
   *     magnitude `producedMagnitude`.
   **/
  // def aggregate(tensor: this.Tensor, reduceOrders: Int, produceMagnitude: Array[Long], transform_f: this.TransformFunction): this.TensorExpr[this.Tensor] = ???

  /**
   * Provides a way for an expression to result in "nothing", relying completely
   * on interpreter-specific operation side-effects.
   **/
  def unit(): this.TensorExpr[Unit] = Free.liftF(this.Unit)

  /**
   * Invert the dimensions of a tensor. For example, for an input 4-D tensor,
   * the result would be seentially the same tensor in
   * (_W, _Z, _Y, _X) orientation. Useful sometimes to prepare tensors for reducing
   * or aggregating, or for re-establishing tensor shape after reducing or aggregating.
   **/
  def invert(tensor: this.Tensor): TensorExpr[this.Tensor] = {
    val expr: this.TensorExpr[this.Tensor] = Free.pure(tensor)
    (0 to ((tensor.order - 1) / 2)).map(_.asInstanceOf[Dimension]).toList.foldLeft(expr) { case (acc, d) =>
        acc.flatMap(pivot(_, d, (tensor.order - d - 1).asInstanceOf[Dimension]))
    }
  }

  /**
   * 2-D matrix multiplcation. In tensor terms, the solution is:
   * * pivot operand 1 (_X, _Z)
   *   * broadcast in _X direction
   * * pivot operand 2 (_Y, _Z)
   *   * broadcast in _Y direction
   * * (At this point we have two order 3 tensors of equal magnitude)
   * * Join the two tensor in _W dimension
   * * Invert and reduce by multiplication in _X, and then summation in _X,
   *   and finally invert once more to get result
   *
   * if operation 1 is ***P***x***M*** and operand 2 is ***N***x***P***, the
   * solution in the tensor algebra is to create a MxNxPx2 tensor. This correctly
   * matches the rows of operand 1 with the columns of operand 2 in Px2 pairs.
   * Then we simply reduce by multiplcation and summation.
   **/
  def matmult2D(op1: this.Tensor, op2: this.Tensor): this.TensorExpr[this.Tensor] = {
    if(op1.magnitude(_X) != op2.magnitude(_Y)) throw new IllegalArgumentException("op1.magnitude(_X) must equal op2.magnitude(_Y)")

    for(e11  <- pivot(op1, _X, _Z);
        e12  <- broadcast(e11, _X, op2.magnitude(_X));
        e21  <- pivot(op2, _Y, _Z);
        e22  <- broadcast(e21, _Y, op1.magnitude(_Y));
        j0   <- join(_W, e12, e22);
        j1   <- invert(j0);
        r0   <- reduce(j1, 1, PRODUCT);
        r1   <- reduce(r0, 1, SUM);
        res  <- invert(r1)) yield {
      res
    }
  }

  /**
   * N-dimensional cross-correlation operation. The source tensor is
   * effectively dilated with 0 values before applying the kernel. Requirements:
   * * `tensor` and `kernel` must be the same order
   * * `kernel` should have odd-sized magnitude in all dimensions
   *
   * Typically `kernel` is smaller than `tensor` in all dimensions -- usually
   * much smaller. For example, magnitude 3 in each dimension can be used to
   * implement N-dimensional Jacobian operator.
   **/
  def crossCorrelate(tensor: this.Tensor, kernel: this.Tensor): this.TensorExpr[this.Tensor] = {
    //...ensure tensor and kernel dimensionality match, and kernel is odd-sized...
    if(tensor.order != kernel.order) throw new IllegalArgumentException("Tensor and kernel must have matching dimensionality")
    if(kernel.elementSize % 2 != 1) throw new IllegalArgumentException("Kernel elementSize must be odd")

    //...record the next 2 higher dimensions, which we will use to broadcast the
    //   kernel and join translated copies of the tensor...
    val d1 = tensor.order.asInstanceOf[Dimension]
    val d2 = (d1 + 1).asInstanceOf[Dimension]

    //...create "kernel.elementSize" tranlated versions of tensor and join
    //   them in the next-higher dimension -- each 1x1x(kernel.elementSize) slice is
    //   a copy of the neighborhood at each (x, y), backfilled with 0 values...
    def rec(mags: List[Long]): List[List[Long]] = mags match {
      case n :: Nil =>
        ((n/2) to -(n/2) by -1).toList.map(List(_))
      case n :: rest =>
        val children = rec(rest)
        for(c <- children;
            i <- ((n/2) to -(n/2) by -1) toList) yield { i +: c }
      //...including the Nil case to keep compiler happy. This case cannot occur because kernel.order must be at least 1.
      case Nil => List()
    }

    val translations: this.TensorExpr[this.Tensor] = {
      val translationOffsets = rec(kernel.magnitude toList)
      translationOffsets.map(offsets => translate(tensor, offsets toArray)) match {
        case head :: rest => rest.foldLeft(head)({ case (accExpr, expr) =>
          accExpr.flatMap(acc =>
            expr.flatMap(t => join(d1, acc, t)))
        })
        //...including the Nil case to keep compiler happy. This case cannot occur because kernel.order must be at least 1.
        case Nil => unit.map(_ => null.asInstanceOf[this.Tensor])
      }
    }

    //...reshape and broadcast kernel to match shape of translations. That is:
    //   * have extent only in dimension d1 (i.e., unitary in
    //   all lower dimensions)
    //   * Broadcast this to shape of source tensor in all lower dimensions...
    val reshapedKernelMag = {
      val m = Array.fill[Long](d1+1)(1L)
      m(d1) = kernel.elementSize
      m
    }
    val liftedKernel = reshape(kernel, reshapedKernelMag)
    val broadcastKernel = (0 to (d1 - 1)).map(_.asInstanceOf[Dimension])
        .foldLeft(liftedKernel)({ case (k, d) => k.flatMap(broadcast(_, d, tensor.magnitude(d))) })

    //...join translations and broadcast kernel. Must then invert dimensions so that
    //   the resultant tensor is 2x(kernel.elementSize) in the _X and _Y dimensions
    //   to prepare for reducing...
    val joined =
      for(ts <- translations;
          bk <- broadcastKernel;
          j  <- join(d2, ts, bk);
          inverted <- invert(j)) yield { inverted }

    //...reduce in first dimension by multiplication, and then reduce one more dimension
    //   (magnitude kernel.elementSize) by addition. Invert the result to put
    //   dimensions back into original tensor orientation...
    joined.flatMap(reduce(_, 1, PRODUCT))
      .flatMap(reduce(_, 1, SUM))
      .flatMap(invert(_))
  }

  /**
   * N-dimensional convolution operation. Requirements:
   * * `tensor` and `kernel` must be the same order
   * * `kernel` should have odd-sized magnitude in all dimensions
   *
   * Typically `kernel` is smaller than `tensor` in all dimensions -- usually
   * much smaller (e.g., magnitude 3 in each dimension).
   **/
  // def convolve(tensor: this.Tensor, kernel: this.Tensor): this.TensorExpr[this.Tensor] = {
  //   /*
  //    * Convolution is cross-correlation with the kernel reversed in all dimensions.
  //    */
  //   val reversedKernel = (_Y to (kernel.order - 1)).map(_.asInstanceOf[Dimension])
  //       .foldLeft(reverse(kernel, _X))({ case (k, dim) => k.flatMap(reverse(_, dim)) })
  //   reversedKernel.flatMap(crossCorrelate(tensor, _))
  // }

  // TBD: The following is saved code to be used to implement split, a function
  // that should divide an original tensor into a grid of slices.
  //
  // val splitMagnitude = {
  //   val splitsMap: Map[Dimension, Long] =
  //     splitDimensions zipWith (splitStepping) toMap
  //   //...calc original magnitudes, with split ones replaced by correct fractional size
  //   val allMags =
  //     for(d <- 0 to tensor.order - 1;
  //         m = tensor.magnitude(d)) yield {
  //       splitsMap.getOrElse(d, m)
  //     }
  //   //...truncate list of magnitudes to remove mags reduced to 1
  //   val lastRemovedDimension = allMags.lastIndexOf(1)
  //   allMags.take(lastRemovedDimension)
  // }
  //
  // val splitSteppingIndexes: List[List[Long]] = {
  //   @tailrec
  //   def rec(mags: List[Long]): List[List[Long]] = mags match {
  //     case Nil => List()
  //     case n :: rest =>
  //       val children = rec(rest)
  //       for(i <- 0 to n;
  //           c <- children) yield { n +: c }
  //   }
  //
  //   val splitDimensionCounts: List[Long] =
  //     for((m, d) <- (tensor.magnutide zipWithIndex)) yield {
  //       (m / splitMagnitude(d))
  //     } toList
  //
  //   rec(splitDimensionCounts)
  // }

  //...generating an Array of SplitTensor objects, each one with the same
  //   magnitude and stepping of the original tensor, and each one with a
  //   unique stepping index
  // splitSteppingIndexes.map((steppingIndex: List[Long]) =>
  //     SplitTensor(tensor, splitMagnitude, splitStepping, steppingIndex.toArray))
  //   .toArray


}
