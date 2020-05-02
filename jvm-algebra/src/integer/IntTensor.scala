package bmaso.tensoralg.jvm.integer

import scala.language.postfixOps

import scala.math.max

import bmaso.tensoralg.abstractions.{Tensor => abstract_Tensor, Dimension}

sealed trait IntTensor extends abstract_Tensor {
  def valueAt(index: Array[Long], startingAt: Long = 0): Int
  def valueAt1D(_1dIdx: Long): Int = {
    var __1dIdx = _1dIdx
    val idx = Array.fill[Long](order)(0)
    for(d <- 0 to (order - 1);
        m = magnitude(d)) {
      idx(d) = __1dIdx % m
      __1dIdx = __1dIdx / m
    }
    valueAt(idx)
  }
  def magnitudeIn(d: Dimension): Long =
    if(d >= magnitude.length) 1L else magnitude(d)

  override def toString(): String =
    s"${this.getClass.getName}(magnitude = ${magnitude.toList})"
}

/**
 * A tensor whose element values are taken from a backing array of values.
 **/
case class IntArrayTensor(arr: Array[Int], override val magnitude: Array[Long], offset: Int)
    extends IntTensor {
  if(arr.length - offset < this.elementSize) throw new IllegalArgumentException("Backing array size is too small for elementSize")
  if(this.elementSize >= Int.MaxValue) throw new IllegalArgumentException("Magnitude is too large for this tensor implementation; elementSize must be < Int.MaxValue")

  override def valueAt(index: Array[Long], startingAt: Long = 0L): Int = {
    for(ii <- startingAt.toInt to (index.length - 1)) if(index(ii) < 0L || (index(ii) > 0L && ((ii - startingAt.toInt) >= magnitude.length || index(ii) >= magnitude(ii - startingAt.toInt))))
      throw new IllegalArgumentException("Index value is out of range")
    var idx = 0L
    var multiplier = 1L
    for(ii <- startingAt.toInt to (index.length - 1)) {
      idx += index(ii) * multiplier
      multiplier *= (if (index(ii) == 0 && (ii - startingAt.toInt) >= magnitude.length) 1L else magnitude(ii - startingAt.toInt))
    }
    arr((idx + offset).toInt)
  }

  def copySlice(tensor: IntTensor, sliceRange: Array[(Long, Long)]): Unit = ???
  def setValueAt(v: Int, index: Array[Long]): Unit = ???
}

/**
 * A tensor "shifted" in one or more dimensions. Negative or
 * positive offsets cause the source tensor to be "truncated" in that
 * dimension.
 **/
case class TranslateTensor(tensor: IntTensor, offsets: Array[Long])
    extends IntTensor {
  /*
   * A couple odd cases to consider: offsets is longer than source magnitude,
   * offsets is shorter than source magnitude. The expected case is that
   * offsets is equal to magnitude. This code correctly handles all three cases:
   * * When offsets is longer, we want new magnitude to match and be 1 in additionally-
   *   introduced dimensions
   * * When offsets is shorter, we want to assume offsetvaue is 0 in the dimensions
   *   not referenced.
   * * When offsets are the same, we add offset to base magnitude.
   */
  override lazy val magnitude = {
    val offs = Array.fill[Long](max(tensor.order, offsets.length))(0)
    Array.copy(offsets, 0, offs, 0, offsets.length)
    val mag = offs.map(_ + 1)
    Array.copy(tensor.magnitude, 0, mag, 0, tensor.magnitude.length)
    mag
  }
  override def valueAt(index: Array[Long], startingAt: Long = 0): Int = {
    val translatedIndex = Array.tabulate[Long](order)((n: Int) => index(startingAt.toInt + n))
    var oob = false
    for(d <- 0 to order - 1) {
      translatedIndex(d) -= (if (d >= offsets.length) 0 else (offsets(d).toInt))
      if(translatedIndex(d) < 0 || translatedIndex(d) >= magnitude(d)) oob = true
    }
    if(oob) 0 else tensor.valueAt(Array.copyAs[Long](translatedIndex, translatedIndex.length))
  }
}

/**
 * A tensor which duplications the source tensor in the target dimension.
 *
 * Assumption is that magnitude is a multiple of source tensor magnitude
 * in same dimension. Code which constructs this instance must make assure
 * this invariant.
 **/
case class BroadcastTensor(tensor: IntTensor, dimension: Dimension, _magnitude: Long)
    extends IntTensor {
  override lazy val magnitude: Array[Long] = {
    if(_magnitude == 1) tensor.magnitude
    else {
      val ret = Array.fill[Long](max(tensor.magnitude.length, dimension+1))(1L)
      Array.copy(tensor.magnitude, 0, ret, 0, tensor.magnitude.length)
      ret(dimension) *= _magnitude
      ret
    }
  }
  override def valueAt(index: Array[Long], startingAt: Long = 0): Int = {
    val ret = Array.copyAs[Long](index, index.length)
    ret(dimension) = ret(dimension) % tensor.magnitudeIn(dimension)
    tensor.valueAt(ret, startingAt)
  }
}

/**
 * A tensor which is a "slice" of the original. The `sliceRange`
 * must provide values for all dimensions of the original tensor. The resultant
 * magnitude array is reduced in size to remove trailing "1" magnitudes.
 **/
case class SliceTensor(tensor: IntTensor, sliceRange: Array[(Long, Long)])
    extends IntTensor {
  override lazy val magnitude: Array[Long] =
    sliceRange
      .map({ case (_, len) => len})
      .reverse.dropWhile(_ == 1) //...drop all final dimensions with magnitude 1...
      .reverse
  override def valueAt(index: Array[Long], startingAt: Long = 0): Int = {
    val shiftedIndex = Array.fill[Long](tensor.order)(0)
    Array.copy(index, 0, shiftedIndex, 0, index.length)
    for(ii <- 0 to (shiftedIndex.length - 1)) {
      shiftedIndex(ii) += sliceRange(ii)._1
    }
    tensor.valueAt(shiftedIndex)
  }
}

/**
 * Describes a tensor the is a reindexing of the elements of another tensor.
 * The dimensionality and magnitude of this tensor may be different than the
 * original. The `elementSize` must be the same as the source tensor.
 **/
case class ReshapeTensor(tensor: IntTensor, override val magnitude: Array[Long])
    extends IntTensor {
   override def valueAt(index: Array[Long], startingAt: Long = 0): Int = {
     var idx = 0L
     var multiplier = 1L
     for(ii <- startingAt.toInt to (index.length - 1)) {
       idx += index(ii) * multiplier
       multiplier *= (if (index(ii) == 0 && (ii - startingAt.toInt) >= magnitude.length) 1L else magnitude(ii - startingAt.toInt))
     }
     tensor.valueAt1D(idx)
   }
}

/**
 * A tensor that made from the "stacking" of one or more source tensors
 * in the `joiningDimension`. The source tensors may have non-unitary size in the
 * joining dimension. Note that if the source tensors definitely have unitary
 * size in the joining dimension, then the `StackTensor` is a more efficient
 * implementation for joining -- both for in-memory size and for element value
 * access.
 **/
case class JoinTensor(tensors: Array[IntTensor], joiningDimension: Dimension)
    extends IntTensor {
  /*
   * It is assumed that the magnitude of all tensors is the same in all but
   * the joining dimension -- they may or may not be the same in the joining dimension.
   * Note the joining dimensions may be far later than the existing dimensions;
   * e.g., joining two 2x2 tensors in the _AA dimension. The following calculation
   * generates a magnitude array of the correct size, padded with 1's as necessary,
   * whose value in the joining dimension is the sum of the backing tensors.
   */
  override lazy val magnitude = {
    val extendedTensorMags =
      for(t <- tensors) yield {
        val ex = Array.fill[Long](max(t.order, joiningDimension+1))(1)
        Array.copy(t.magnitude, 0, ex, 0, t.order)
        ex
      }
    extendedTensorMags(0)(joiningDimension) = extendedTensorMags.map(_.apply(joiningDimension)).sum
    extendedTensorMags(0)
  }

  /**
   * For internal use when mapping indexes in the joining dimension to specific
   * constituent tensors and their respective magnitudes in the joining dimension.
   * stored in reverse order so that "dropWhile" on this list does not cause
   * a new list to be constructed.
   **/
  lazy val joinDimensionOffsets =
    tensors.map(_.magnitudeIn(joiningDimension)).scan(0L)(_ + _).init.toList.reverse

  override def valueAt(index: Array[Long], startingAt: Long = 0): Int = {
    val idx = Array.copyAs[Long](index, index.length)
    val offsets = joinDimensionOffsets.dropWhile(_ > idx(joiningDimension + startingAt.toInt))
    //^^^ note: joinDimensionOffsets is stored in reverse order so this dropWhile
    //    does not create new list instance
    idx(joiningDimension + startingAt.toInt) = idx(joiningDimension + startingAt.toInt) - offsets.head
    tensors(offsets.length - 1).valueAt(idx, startingAt)
  }
}

/**
 * A tensor that made from the "stacking" of one or more source tensors
 * in the `joiningDimension`. The source tensors must have unitary size in the
 * joining dimension. The implementation utilizes this assumption, and is able
 * to be smaller in-memory as well as aster for element value access.

 *
 * Note there are an assumptions, which should be asserted prior to construction of an
 * instance:
 * * the magnitudes of the constituent tensors are equal in all dimensions
 * * all constituent tensors have unitary magnitude in the joining dimension
 **/
case class StackTensor(tensors: Array[IntTensor], joiningDimension: Dimension)
    extends IntTensor {
  override lazy val magnitude = {
    val extendedTensorMags =
      for(t <- tensors) yield {
        val ex = Array.fill[Long](max(t.order, joiningDimension+1))(1)
        Array.copy(t.magnitude, 0, ex, 0, t.order)
        ex
      }
    extendedTensorMags(0)(joiningDimension) = tensors.length
    extendedTensorMags(0)
  }
  override def valueAt(index: Array[Long], startingAt: Long = 0): Int = {
    val idx = Array.copyAs[Long](index, index.length)
    idx(joiningDimension + startingAt.toInt) = 0
    tensors(index((joiningDimension + startingAt).toInt).toInt).valueAt(idx, startingAt)
  }
}

/**
 * A tensor constructed from a source tensor by reversing the indexes
 * of element values along a single dimension.
 **/
 case class ReverseTensor(tensor: IntTensor, dimension: Dimension)
     extends IntTensor {
   override def magnitude = tensor.magnitude
   override def valueAt(index: Array[Long], startingAt: Long = 0): Int =
     if((index.length - startingAt) >= dimension) {
       val idx = Array.copyAs[Long](index, index.length - startingAt.toInt)
       idx(dimension) = tensor.magnitude(dimension) - 1 - idx(dimension)
       tensor.valueAt(idx, 0)
     } else
       tensor.valueAt(index)
 }

 /**
  * A tensor constructed from a source tensor by exchanging two dimensions.
  **/
 case class PivotTensor(tensor: IntTensor, dim1: Dimension, dim2: Dimension)
     extends IntTensor {
   override lazy val magnitude = {
     val mag = Array.fill[Long](max(tensor.order, max(dim1, dim2) + 1))(1)
     Array.copy(tensor.magnitude, 0, mag, 0, tensor.order)
     val swap = mag(dim1)
     mag(dim1) = mag(dim2)
     mag(dim2) = swap
     mag
   }
   override def valueAt(index: Array[Long], startingAt: Long = 0): Int = {
     if(dim1 == dim2) tensor.valueAt(index)
     else {
       val idx: Array[Long] = {
         if(index.length - startingAt > dim1 && index.length - startingAt > dim2) {
           val ret = Array.fill[Long](index.length - startingAt.toInt)(0)
           Array.copy(index, startingAt.toInt, ret, 0, index.length - startingAt.toInt)
           ret
         } else {
           val ret = Array.fill[Long](max(dim1, dim2) + 1)(0L)
           Array.copy(index, startingAt.toInt, ret, 0, index.length)
           ret
         }
       }
       val swap = idx(dim1)
       idx(dim1) = idx(dim2)
       idx(dim2) = swap
       tensor.valueAt(idx)
     }
   }
 }

 /**
  * A tensor constructed by applying an (Int) => Int mapping from a source
  * tensor's elements.
  **/
 case class MapTensor(tensor: IntTensor, f: (Int) => Int) extends IntTensor {
   override def magnitude = tensor.magnitude
   override def valueAt(index: Array[Long], startingAt: Long = 0): Int = f(tensor.valueAt(index, startingAt))
 }

 /**
  * A tensor constructed by reducing the first `reduceOrders` dimensions using the
  * `reduce_f` reducing function. For example, a 3x4x5x6 tensor with 2-order reduction
  * will yield a 5x6 tensor.
  *
  * When a value is requested, a slice is made of the source tensor corresponding
  * to the element index. The slice ranges are full `(0, full magnitude)` in all
  * dimensions The slice is fed to the reduce function, and the result
  * value is what is returned.
  **/
 case class ReduceTensor(tensor: IntTensor, reduceOrders: Int, f: (IntTensor) => Int)
    extends IntTensor {
   override lazy val magnitude = {
     val ret = tensor.magnitude.drop(reduceOrders)
     if(ret.isEmpty) Array(1) else ret
   }
   override def valueAt(index: Array[Long], startingAt: Long = 0): Int = {
     val sliceRanges = tensor.magnitude.take(reduceOrders).map(m => (0L, m)) :++ index.map(i => (i, 1L))
     val slice = SliceTensor(tensor, sliceRanges)
     f(slice)
   }

 }
